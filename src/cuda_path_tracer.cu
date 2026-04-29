#include "lt/renderer.h"

#if LT_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace lt {
namespace {

struct GpuMaterial {
    Vec3 albedo;
    int brdf_model = 0;
    float roughness = 0.5f;
    float metallic = 0.0f;
    int texture_index = -1;
};

struct GpuTexture {
    int width = 0;
    int height = 0;
    cudaTextureObject_t object = 0;
};

struct GpuTriangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
    Vec3 centroid;
    Vec3 emission;
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    int material = 0;
    int mesh = -1;
};

struct GpuBvhNode {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;
};

struct GpuMeshInstance {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int bvh_root = -1;
    int mesh = -1;
};

struct GpuScene {
    Camera camera;
    int material_count = 0;
    int triangle_count = 0;
    int bvh_node_count = 0;
    int tlas_node_count = 0;
    int mesh_instance_count = 0;
    int use_two_level = 0;
    int light_count = 0;
    int texture_count = 0;
    int environment_texture = -1;
    Vec3 environment_color = {1.0f, 1.0f, 1.0f};
    float environment_strength = 1.0f;
    bool environment_constant = false;
    GpuMaterial* materials = nullptr;
    GpuTexture* textures = nullptr;
    GpuTriangle* triangles = nullptr;
    int* triangle_indices = nullptr;
    int* light_indices = nullptr;
    GpuBvhNode* bvh_nodes = nullptr;
    GpuMeshInstance* mesh_instances = nullptr;
    int* mesh_instance_indices = nullptr;
    GpuBvhNode* tlas_nodes = nullptr;
};

struct PackedGpuScene {
    GpuScene scene;
    std::vector<GpuMaterial> materials;
    std::vector<GpuTexture> textures;
    std::vector<GpuTriangle> triangles;
    std::vector<int> triangle_indices;
    std::vector<int> light_indices;
    std::vector<GpuBvhNode> bvh_nodes;
    std::vector<GpuMeshInstance> mesh_instances;
    std::vector<int> mesh_instance_indices;
    std::vector<GpuBvhNode> tlas_nodes;
};

struct GpuHit {
    float t = kInfinity;
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    int material = -1;
    int mesh = -1;
    int triangle = -1;
    Vec3 emission;
    bool front_face = true;
};

struct GpuPrimaryHit {
    Ray ray;
    GpuHit hit;
    int valid = 0;
};

struct GpuMaterialSample {
    Vec3 direction;
    Vec3 weight;
    float pdf = 0.0f;
    bool delta = false;
};

__host__ __device__ Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__host__ __device__ Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ Vec3 mul(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
__host__ __device__ Vec3 mul(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__host__ __device__ Vec2 add(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
__host__ __device__ Vec2 mul(Vec2 a, float s) { return {a.x * s, a.y * s}; }
__host__ __device__ Vec3 divv(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
__host__ __device__ float ddot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ Vec3 dcross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ Vec3 dnormalize(Vec3 v) {
    const float len = sqrtf(ddot(v, v));
    return len > 0.0f ? divv(v, len) : Vec3{};
}
__host__ __device__ float dclamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}
__host__ __device__ float wrap01_gpu(float v) {
    v = v - floorf(v);
    return v < 0.0f ? v + 1.0f : v;
}
__host__ __device__ int iclamp_gpu(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__host__ __device__ Vec3 dlerp(Vec3 a, Vec3 b, float t) {
    return add(mul(a, 1.0f - t), mul(b, t));
}
__host__ __device__ uint32_t rng_next(uint32_t& state) {
    state = state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
__host__ __device__ float rng_float(uint32_t& state) {
    return static_cast<float>((rng_next(state) >> 8) * (1.0 / 16777216.0));
}
__device__ Vec3 cosine_sample(float u1, float u2) {
    const float r = sqrtf(u1);
    const float phi = 2.0f * kPi * u2;
    return {r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1))};
}
__device__ Vec3 to_world_gpu(Vec3 local, Vec3 n) {
    const Vec3 up = fabsf(n.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = dnormalize(dcross(up, n));
    const Vec3 bitangent = dcross(n, tangent);
    return add(add(mul(tangent, local.x), mul(bitangent, local.y)), mul(n, local.z));
}
__device__ uint32_t rgba8_gpu(Vec3 color) {
    color.x = dclamp(color.x, 0.0f, 1.0f);
    color.y = dclamp(color.y, 0.0f, 1.0f);
    color.z = dclamp(color.z, 0.0f, 1.0f);
    const uint32_t r = static_cast<uint32_t>(dclamp(powf(color.x, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f + 0.5f);
    const uint32_t g = static_cast<uint32_t>(dclamp(powf(color.y, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f + 0.5f);
    const uint32_t b = static_cast<uint32_t>(dclamp(powf(color.z, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f + 0.5f);
    return 0xff000000u | (r << 16u) | (g << 8u) | b;
}

__device__ Ray camera_ray(const Camera& camera, int x, int y, int width, int height, uint32_t& rng) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float half_height = tanf(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = dnormalize(sub(camera.target, camera.position));
    const Vec3 right = dnormalize(dcross(forward, camera.up));
    const Vec3 up = dcross(right, forward);
    const float u = ((static_cast<float>(x) + rng_float(rng)) / static_cast<float>(width) * 2.0f - 1.0f) * half_width;
    const float v = (1.0f - (static_cast<float>(y) + rng_float(rng)) / static_cast<float>(height) * 2.0f) * half_height;
    return {camera.position, dnormalize(add(add(forward, mul(right, u)), mul(up, v)))};
}

__device__ Ray camera_ray_center(const Camera& camera, int x, int y, int width, int height) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float half_height = tanf(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = dnormalize(sub(camera.target, camera.position));
    const Vec3 right = dnormalize(dcross(forward, camera.up));
    const Vec3 up = dcross(right, forward);
    const float u = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f) * half_width;
    const float v = (1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.0f) * half_height;
    return {camera.position, dnormalize(add(add(forward, mul(right, u)), mul(up, v)))};
}

__device__ bool intersect_triangle(const GpuTriangle& tri, const Ray& ray, float& t, float& u, float& v) {
    const Vec3 edge1 = sub(tri.v1, tri.v0);
    const Vec3 edge2 = sub(tri.v2, tri.v0);
    const Vec3 p = dcross(ray.direction, edge2);
    const float det = ddot(edge1, p);
    if (fabsf(det) < 1.0e-7f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 s = sub(ray.origin, tri.v0);
    u = inv_det * ddot(s, p);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3 q = dcross(s, edge1);
    v = inv_det * ddot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    t = inv_det * ddot(edge2, q);
    return t > 0.001f;
}

__device__ bool intersect_aabb_gpu(Vec3 bounds_min, Vec3 bounds_max, const Ray& ray, float max_t, float* out_tmin = nullptr) {
    float tmin = 0.001f;
    float tmax = max_t;
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float direction = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float lo = axis == 0 ? bounds_min.x : axis == 1 ? bounds_min.y : bounds_min.z;
        const float hi = axis == 0 ? bounds_max.x : axis == 1 ? bounds_max.y : bounds_max.z;
        if (fabsf(direction) < 1.0e-8f) {
            if (origin < lo || origin > hi) {
                return false;
            }
            continue;
        }
        const float inv = 1.0f / direction;
        float t0 = (lo - origin) * inv;
        float t1 = (hi - origin) * inv;
        if (inv < 0.0f) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax < tmin) {
            return false;
        }
    }
    if (out_tmin) {
        *out_tmin = tmin;
    }
    return true;
}

__device__ bool intersect_bvh_gpu(const GpuScene& scene, const GpuBvhNode* nodes, int node_count, const int* indices, int index_count, const Ray& ray, int root, GpuHit& hit) {
    bool found = false;
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= node_count) {
            continue;
        }
        const GpuBvhNode node = nodes[node_index];
        if (!intersect_aabb_gpu(node.bounds_min, node.bounds_max, ray, hit.t)) {
            continue;
        }

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= index_count) {
                    continue;
                }
                const int tri_index = indices[index_offset];
                if (tri_index < 0 || tri_index >= scene.triangle_count) {
                    continue;
                }
                const GpuTriangle& tri = scene.triangles[tri_index];
                if (tri.material < 0 || tri.material >= scene.material_count) {
                    continue;
                }
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(tri, ray, t, u, v) && t < hit.t) {
                    const Vec3 shading_normal = dnormalize(add(add(mul(tri.n0, 1.0f - u - v), mul(tri.n1, u)), mul(tri.n2, v)));
                    hit.t = t;
                    hit.position = add(ray.origin, mul(ray.direction, t));
                    const Vec3 hit_normal = ddot(shading_normal, shading_normal) > 0.0f ? shading_normal : tri.normal;
                    hit.front_face = ddot(tri.normal, ray.direction) < 0.0f;
                    hit.normal = ddot(hit_normal, ray.direction) < 0.0f ? hit_normal : mul(hit_normal, -1.0f);
                    hit.uv = add(add(mul(tri.uv0, 1.0f - u - v), mul(tri.uv1, u)), mul(tri.uv2, v));
                    hit.material = tri.material;
                    hit.mesh = tri.mesh;
                    hit.triangle = tri_index;
                    hit.emission = tri.emission;
                    found = true;
                }
            }
        } else {
            float left_t = kInfinity;
            float right_t = kInfinity;
            const bool hit_left = node.left >= 0 && node.left < node_count &&
                intersect_aabb_gpu(nodes[node.left].bounds_min, nodes[node.left].bounds_max, ray, hit.t, &left_t);
            const bool hit_right = node.right >= 0 && node.right < node_count &&
                intersect_aabb_gpu(nodes[node.right].bounds_min, nodes[node.right].bounds_max, ray, hit.t, &right_t);
            if (hit_left && hit_right) {
                const int near_child = left_t <= right_t ? node.left : node.right;
                const int far_child = left_t <= right_t ? node.right : node.left;
                if (stack_size < 64) stack[stack_size++] = far_child;
                if (stack_size < 64) stack[stack_size++] = near_child;
            } else if (hit_left) {
                if (stack_size < 64) stack[stack_size++] = node.left;
            } else if (hit_right) {
                if (stack_size < 64) stack[stack_size++] = node.right;
            }
        }
    }
    return found;
}

__device__ bool intersect_blas_gpu(const GpuScene& scene, const Ray& ray, int root, GpuHit& hit) {
    return intersect_bvh_gpu(scene, scene.bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, root, hit);
}

__device__ bool intersect_gpu(const GpuScene& scene, const Ray& ray, GpuHit& hit) {
    if (scene.use_two_level == 0) {
        return scene.bvh_node_count > 0 &&
            intersect_bvh_gpu(scene, scene.bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, 0, hit);
    }
    if (scene.tlas_node_count <= 0 || scene.mesh_instance_count <= 0) {
        return false;
    }

    bool found = false;
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= scene.tlas_node_count) {
            continue;
        }
        const GpuBvhNode node = scene.tlas_nodes[node_index];
        if (!intersect_aabb_gpu(node.bounds_min, node.bounds_max, ray, hit.t)) {
            continue;
        }

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= scene.mesh_instance_count) {
                    continue;
                }
                const int instance_index = scene.mesh_instance_indices[index_offset];
                if (instance_index < 0 || instance_index >= scene.mesh_instance_count) {
                    continue;
                }
                const GpuMeshInstance instance = scene.mesh_instances[instance_index];
                if (instance.bvh_root >= 0 &&
                    intersect_aabb_gpu(instance.bounds_min, instance.bounds_max, ray, hit.t)) {
                    found = intersect_blas_gpu(scene, ray, instance.bvh_root, hit) || found;
                }
            }
        } else {
            float left_t = kInfinity;
            float right_t = kInfinity;
            const bool hit_left = node.left >= 0 && node.left < scene.tlas_node_count &&
                intersect_aabb_gpu(scene.tlas_nodes[node.left].bounds_min, scene.tlas_nodes[node.left].bounds_max, ray, hit.t, &left_t);
            const bool hit_right = node.right >= 0 && node.right < scene.tlas_node_count &&
                intersect_aabb_gpu(scene.tlas_nodes[node.right].bounds_min, scene.tlas_nodes[node.right].bounds_max, ray, hit.t, &right_t);
            if (hit_left && hit_right) {
                const int near_child = left_t <= right_t ? node.left : node.right;
                const int far_child = left_t <= right_t ? node.right : node.left;
                if (stack_size < 64) stack[stack_size++] = far_child;
                if (stack_size < 64) stack[stack_size++] = near_child;
            } else if (hit_left) {
                if (stack_size < 64) stack[stack_size++] = node.left;
            } else if (hit_right) {
                if (stack_size < 64) stack[stack_size++] = node.right;
            }
        }
    }
    return found;
}

__host__ __device__ bool has_light_emission_gpu(Vec3 emission) {
    return emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
}

__host__ __device__ Vec3 emitted_radiance_gpu(const GpuTriangle& light, Vec3 ray_direction_to_light) {
    return has_light_emission_gpu(light.emission) && ddot(light.normal, mul(ray_direction_to_light, -1.0f)) > 0.0f ? light.emission : Vec3{};
}

__device__ Vec3 fresnel_schlick_gpu(float cos_theta, Vec3 f0) {
    const float f = powf(dclamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
    return add(f0, mul(sub(Vec3{1.0f, 1.0f, 1.0f}, f0), f));
}

__device__ Vec3 sample_texture_gpu(const GpuScene& scene, int texture_index, Vec2 uv) {
    if (texture_index < 0 || texture_index >= scene.texture_count) {
        return {1.0f, 1.0f, 1.0f};
    }
    const GpuTexture texture = scene.textures[texture_index];
    if (texture.width <= 0 || texture.height <= 0) {
        return {1.0f, 1.0f, 1.0f};
    }
    if (texture.object == 0) {
        return {1.0f, 1.0f, 1.0f};
    }
    const float4 sample = tex2D<float4>(texture.object, wrap01_gpu(uv.x), wrap01_gpu(1.0f - uv.y));
    return {sample.x, sample.y, sample.z};
}

__device__ Vec3 environment_radiance_gpu(const GpuScene& scene, Vec3 direction) {
    if (scene.environment_texture >= 0) {
        const float u = atan2f(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
        const float v = acosf(dclamp(direction.y, -1.0f, 1.0f)) / kPi;
        return mul(mul(scene.environment_color, sample_texture_gpu(scene, scene.environment_texture, {u, 1.0f - v})), scene.environment_strength);
    }
    if (scene.environment_constant) {
        return mul(scene.environment_color, scene.environment_strength);
    }
    const float t = 0.5f * (direction.y + 1.0f);
    return mul(mul(add(mul(Vec3{0.02f, 0.025f, 0.035f}, 1.0f - t), mul(Vec3{0.32f, 0.45f, 0.68f}, t)), scene.environment_color), scene.environment_strength);
}

__device__ float ggx_distribution_gpu(float ndoth, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(1.0e-6f, kPi * d * d);
}

__device__ float smith_ggx_g1_gpu(float ndotv, float roughness) {
    const float r = roughness + 1.0f;
    const float k = (r * r) * 0.125f;
    return ndotv / fmaxf(1.0e-6f, ndotv * (1.0f - k) + k);
}

__device__ Vec3 evaluate_brdf_gpu(const GpuScene& scene, const GpuMaterial& material, Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) {
    const float ndotv = fmaxf(0.0f, ddot(n, wo));
    const float ndotl = fmaxf(0.0f, ddot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f) {
        return {};
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Lambertian)) {
        return divv(mul(material.albedo, sample_texture_gpu(scene, material.texture_index, uv)), kPi);
    }
    if (material.brdf_model != static_cast<int>(BrdfModel::Principled)) {
        return {};
    }

    const float roughness = dclamp(material.roughness, 0.02f, 1.0f);
    const float metallic = dclamp(material.metallic, 0.0f, 1.0f);
    const Vec3 h = dnormalize(add(wi, wo));
    const float ndoth = fmaxf(0.0f, ddot(n, h));
    const float vdoth = fmaxf(0.0f, ddot(wo, h));
    const Vec3 color = mul(material.albedo, sample_texture_gpu(scene, material.texture_index, uv));
    const Vec3 f0 = dlerp(Vec3{0.04f, 0.04f, 0.04f}, color, metallic);
    const Vec3 f = fresnel_schlick_gpu(vdoth, f0);
    const float d = ggx_distribution_gpu(ndoth, roughness);
    const float g = smith_ggx_g1_gpu(ndotv, roughness) * smith_ggx_g1_gpu(ndotl, roughness);
    const Vec3 specular = mul(f, d * g / fmaxf(1.0e-6f, 4.0f * ndotv * ndotl));
    const Vec3 diffuse = divv(mul(color, 1.0f - metallic), kPi);
    return add(diffuse, specular);
}

__device__ Vec3 reflect_gpu(Vec3 v, Vec3 n) {
    return sub(mul(n, 2.0f * ddot(v, n)), v);
}

__device__ Vec3 refract_gpu(Vec3 v, Vec3 n, float eta) {
    const float cos_theta = fminf(ddot(v, n), 1.0f);
    const Vec3 perpendicular = mul(sub(mul(n, cos_theta), v), eta);
    const float parallel_len2 = fmaxf(0.0f, 1.0f - ddot(perpendicular, perpendicular));
    return dnormalize(sub(perpendicular, mul(n, sqrtf(parallel_len2))));
}

__device__ float fresnel_dielectric_gpu(float cos_theta, float eta) {
    float r0 = (1.0f - eta) / (1.0f + eta);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(dclamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
}

__device__ Vec3 sample_ggx_half_gpu(Vec3 n, float roughness, float u1, float u2) {
    const float a = roughness * roughness;
    const float phi = 2.0f * kPi * u1;
    const float cos_theta = sqrtf((1.0f - u2) / fmaxf(1.0e-6f, 1.0f + (a * a - 1.0f) * u2));
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    return dnormalize(to_world_gpu({sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta}, n));
}

__device__ float material_pdf_gpu(const GpuMaterial& material, Vec3 n, Vec3 wo, Vec3 wi) {
    const float ndotl = fmaxf(0.0f, ddot(n, wi));
    const float ndotv = fmaxf(0.0f, ddot(n, wo));
    if (ndotl <= 0.0f || ndotv <= 0.0f) {
        return 0.0f;
    }
    const float diffuse_pdf = ndotl / kPi;
    if (material.brdf_model == static_cast<int>(BrdfModel::Lambertian)) {
        return diffuse_pdf;
    }
    if (material.brdf_model != static_cast<int>(BrdfModel::Principled)) {
        return 0.0f;
    }
    const float roughness = dclamp(material.roughness, 0.02f, 1.0f);
    const float metallic = dclamp(material.metallic, 0.0f, 1.0f);
    const float spec_prob = dclamp(0.25f + 0.75f * metallic, 0.25f, 1.0f);
    const Vec3 h = dnormalize(add(wi, wo));
    const float ndoth = fmaxf(0.0f, ddot(n, h));
    const float vdoth = fmaxf(0.0f, ddot(wo, h));
    const float specular_pdf = ggx_distribution_gpu(ndoth, roughness) * ndoth / fmaxf(1.0e-6f, 4.0f * vdoth);
    return (1.0f - spec_prob) * diffuse_pdf + spec_prob * specular_pdf;
}

__device__ GpuMaterialSample sample_material_gpu(const GpuScene& scene, const GpuMaterial& material, Vec3 n, Vec3 wo, Vec2 uv, bool front_face, uint32_t& rng) {
    GpuMaterialSample result;
    const Vec3 color = mul(material.albedo, sample_texture_gpu(scene, material.texture_index, uv));
    if (material.brdf_model == static_cast<int>(BrdfModel::Mirror) || material.brdf_model == static_cast<int>(BrdfModel::Conductor)) {
        result.direction = dnormalize(reflect_gpu(wo, n));
        result.weight = material.brdf_model == static_cast<int>(BrdfModel::Mirror) ? Vec3{1.0f, 1.0f, 1.0f} : color;
        result.pdf = 1.0f;
        result.delta = true;
        return result;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Dielectric)) {
        const float ior = dclamp(material.roughness, 1.0f, 3.0f);
        const float eta = front_face ? 1.0f / ior : ior;
        const float cos_theta = fminf(ddot(wo, n), 1.0f);
        const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
        const bool cannot_refract = eta * sin_theta > 1.0f;
        const float reflectance = fresnel_dielectric_gpu(cos_theta, eta);
        result.direction = (cannot_refract || reflectance > rng_float(rng)) ? dnormalize(reflect_gpu(wo, n)) : refract_gpu(wo, n, eta);
        result.weight = color;
        result.pdf = 1.0f;
        result.delta = true;
        return result;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Principled)) {
        const float metallic = dclamp(material.metallic, 0.0f, 1.0f);
        const float spec_prob = dclamp(0.25f + 0.75f * metallic, 0.25f, 1.0f);
        if (rng_float(rng) < spec_prob) {
            const Vec3 h = sample_ggx_half_gpu(n, dclamp(material.roughness, 0.02f, 1.0f), rng_float(rng), rng_float(rng));
            result.direction = dnormalize(reflect_gpu(wo, h));
            if (ddot(result.direction, n) <= 0.0f) {
                result.direction = dnormalize(to_world_gpu(cosine_sample(rng_float(rng), rng_float(rng)), n));
            }
        } else {
            result.direction = dnormalize(to_world_gpu(cosine_sample(rng_float(rng), rng_float(rng)), n));
        }
    } else {
        result.direction = dnormalize(to_world_gpu(cosine_sample(rng_float(rng), rng_float(rng)), n));
    }
    result.pdf = material_pdf_gpu(material, n, wo, result.direction);
    const float ndotl = fmaxf(0.0f, ddot(n, result.direction));
    result.weight = result.pdf > 0.0f ? mul(evaluate_brdf_gpu(scene, material, n, wo, result.direction, uv), ndotl / result.pdf) : Vec3{};
    return result;
}

__device__ Vec3 sample_triangle_area_gpu(const GpuTriangle& tri, uint32_t& rng) {
    const float su = sqrtf(rng_float(rng));
    const float v = rng_float(rng);
    return add(add(mul(tri.v0, 1.0f - su), mul(tri.v1, su * (1.0f - v))), mul(tri.v2, su * v));
}

__device__ float triangle_area_gpu(const GpuTriangle& tri) {
    const Vec3 edge1 = sub(tri.v1, tri.v0);
    const Vec3 edge2 = sub(tri.v2, tri.v0);
    return 0.5f * sqrtf(ddot(dcross(edge1, edge2), dcross(edge1, edge2)));
}

__device__ float light_pdf_solid_angle_gpu(const GpuTriangle& light, Vec3 origin, Vec3 light_point, float light_pmf) {
    const Vec3 to_light = sub(light_point, origin);
    const float dist2 = ddot(to_light, to_light);
    if (dist2 <= 1.0e-8f) {
        return 0.0f;
    }
    const Vec3 light_dir = divv(to_light, sqrtf(dist2));
    const float ldot = fmaxf(0.0f, ddot(mul(light.normal, -1.0f), light_dir));
    const float area = triangle_area_gpu(light);
    return ldot > 0.0f && area > 0.0f && light_pmf > 0.0f ? light_pmf * dist2 / (ldot * area) : 0.0f;
}

__device__ float mis_weight_gpu(float pdf_a, float pdf_b, int heuristic) {
    if (heuristic == static_cast<int>(MisHeuristic::Balance)) {
        return pdf_a / fmaxf(1.0e-8f, pdf_a + pdf_b);
    }
    const float a2 = pdf_a * pdf_a;
    const float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(1.0e-8f, a2 + b2);
}

__device__ Vec3 estimate_direct_gpu(const GpuScene& scene, const GpuHit& hit, const GpuMaterial& material, Vec3 wo, uint32_t& rng, const RenderSettings& settings) {
    Vec3 direct{};
    if (scene.light_count <= 0) {
        return direct;
    }
    const int list_index = iclamp_gpu(static_cast<int>(rng_float(rng) * static_cast<float>(scene.light_count)), 0, scene.light_count - 1);
    const int light_index = scene.light_indices[list_index];
    if (light_index < 0 || light_index >= scene.triangle_count) {
        return direct;
    }
    const GpuTriangle light = scene.triangles[light_index];
    if (!has_light_emission_gpu(light.emission)) {
        return direct;
    }
    const Vec3 light_point = sample_triangle_area_gpu(light, rng);
    const Vec3 to_light = sub(light_point, hit.position);
    const float dist2 = ddot(to_light, to_light);
    if (dist2 <= 1.0e-8f) {
        return direct;
    }
    const float dist = sqrtf(dist2);
    const Vec3 light_dir = divv(to_light, dist);
    const float ndotl = fmaxf(0.0f, ddot(hit.normal, light_dir));
    const float ldot = fmaxf(0.0f, ddot(mul(light.normal, -1.0f), light_dir));
    if (ndotl <= 0.0f || ldot <= 0.0f) {
        return direct;
    }
    GpuHit shadow_hit;
    const Ray shadow_ray{add(hit.position, mul(hit.normal, 0.002f)), light_dir};
    if (intersect_gpu(scene, shadow_ray, shadow_hit)) {
        if (shadow_hit.t < dist - 0.01f || shadow_hit.triangle != light_index) {
            return direct;
        }
    }
    const float light_pmf = 1.0f / static_cast<float>(scene.light_count);
    const float light_pdf = light_pdf_solid_angle_gpu(light, hit.position, light_point, light_pmf);
    const float bsdf_pdf = material_pdf_gpu(material, hit.normal, wo, light_dir);
    const float weight = settings.use_mis ? mis_weight_gpu(light_pdf, bsdf_pdf, static_cast<int>(settings.mis_heuristic)) : 1.0f;
    direct = add(direct, mul(mul(evaluate_brdf_gpu(scene, material, hit.normal, wo, light_dir, hit.uv), light.emission), ndotl * weight / fmaxf(1.0e-8f, light_pdf)));
    return direct;
}

__device__ Vec3 trace_gpu_from_primary(const GpuScene& scene, Ray ray, const GpuPrimaryHit* primary, uint32_t& rng, const RenderSettings& settings) {
    Vec3 radiance{};
    Vec3 throughput{1.0f, 1.0f, 1.0f};
    Vec3 previous_position{};
    float previous_bsdf_pdf = 0.0f;
    bool previous_delta = false;
    for (int bounce = 0; bounce < settings.max_bounces; ++bounce) {
        GpuHit hit;
        if (bounce == 0 && primary) {
            ray = primary->ray;
            if (primary->valid <= 0) {
                radiance = add(radiance, mul(throughput, environment_radiance_gpu(scene, ray.direction)));
                break;
            }
            hit = primary->hit;
        } else if (!intersect_gpu(scene, ray, hit)) {
            radiance = add(radiance, mul(throughput, environment_radiance_gpu(scene, ray.direction)));
            break;
        }
        if (hit.material < 0 || hit.material >= scene.material_count) {
            break;
        }
        const GpuMaterial material = scene.materials[hit.material];
        if (has_light_emission_gpu(hit.emission)) {
            const GpuTriangle light = hit.triangle >= 0 && hit.triangle < scene.triangle_count ? scene.triangles[hit.triangle] : GpuTriangle{};
            const Vec3 emission = emitted_radiance_gpu(light, ray.direction);
            if (bounce == 0) {
                radiance = add(radiance, mul(throughput, emission));
            } else if (previous_delta) {
                radiance = add(radiance, mul(throughput, emission));
            } else if (settings.use_mis && hit.triangle >= 0 && hit.triangle < scene.triangle_count) {
                const float light_pmf = scene.light_count > 0 ? 1.0f / static_cast<float>(scene.light_count) : 0.0f;
                const float light_pdf = light_pdf_solid_angle_gpu(light, previous_position, hit.position, light_pmf);
                radiance = add(radiance, mul(mul(throughput, emission), mis_weight_gpu(previous_bsdf_pdf, light_pdf, static_cast<int>(settings.mis_heuristic))));
            }
            break;
        }
        const Vec3 wo = mul(ray.direction, -1.0f);
        radiance = add(radiance, mul(throughput, estimate_direct_gpu(scene, hit, material, wo, rng, settings)));
        if (bounce >= 3) {
            const float p = dclamp(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng_float(rng) > p) {
                break;
            }
            throughput = divv(throughput, p);
        }
        const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal, wo, hit.uv, hit.front_face, rng);
        if (sample.pdf <= 0.0f || ddot(sample.weight, sample.weight) <= 0.0f) {
            break;
        }
        previous_position = hit.position;
        previous_bsdf_pdf = sample.pdf;
        previous_delta = sample.delta;
        const float offset_side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
        ray = {add(hit.position, mul(hit.normal, 0.001f * offset_side)), sample.direction};
        throughput = mul(throughput, sample.weight);
    }
    return radiance;
}

__device__ GpuPrimaryHit make_primary_hit(const GpuScene& scene, int x, int y, const RenderSettings& settings) {
    GpuPrimaryHit primary;
    primary.ray = camera_ray_center(scene.camera, x, y, settings.width, settings.height);
    primary.valid = intersect_gpu(scene, primary.ray, primary.hit) ? 1 : 0;
    return primary;
}

__global__ void render_kernel(const GpuScene* scene_ptr, RenderSettings settings, Vec3* accumulation, uint32_t* rgba, GpuPrimaryHit* primary_hits, int primary_cache_ready) {
    const GpuScene& scene = *scene_ptr;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) {
        return;
    }
    const int idx = y * settings.width + x;
    uint32_t rng = make_pixel_seed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), settings.frame_index);
    Vec3 sample{};
    GpuPrimaryHit primary;
    const GpuPrimaryHit* primary_ptr = nullptr;
    if (settings.use_primary_hit_cache && primary_hits) {
        if (primary_cache_ready) {
            primary = primary_hits[idx];
        } else {
            primary = make_primary_hit(scene, x, y, settings);
            primary_hits[idx] = primary;
        }
        primary_ptr = &primary;
    }
    for (int s = 0; s < settings.samples_per_pixel; ++s) {
        if (primary_ptr) {
            sample = add(sample, trace_gpu_from_primary(scene, primary.ray, primary_ptr, rng, settings));
        } else {
            sample = add(sample, trace_gpu_from_primary(scene, camera_ray(scene.camera, x, y, settings.width, settings.height, rng), nullptr, rng, settings));
        }
    }
    sample = divv(sample, static_cast<float>(settings.samples_per_pixel));
    accumulation[idx] = add(accumulation[idx], sample);
    rgba[idx] = rgba8_gpu(divv(accumulation[idx], static_cast<float>(settings.frame_index + 1u)));
}

bool size_fits_int(size_t size) {
    return size <= static_cast<size_t>(std::numeric_limits<int>::max());
}

bool use_two_level_accel(AccelerationStructure acceleration_structure) {
    return acceleration_structure == AccelerationStructure::TwoLevel;
}

bool pack_scene(const Scene& scene, const RenderSettings& settings, PackedGpuScene& packed) {
    packed = {};
    GpuScene& gpu = packed.scene;
    gpu.camera = scene.camera;
    gpu.use_two_level = use_two_level_accel(settings.acceleration_structure) ? 1 : 0;
    gpu.environment_color = scene.environment.color;
    gpu.environment_strength = scene.environment.strength;
    gpu.environment_constant = scene.environment.constant;
    if (!size_fits_int(scene.textures.size()) || !size_fits_int(scene.materials.size())) {
        return false;
    }
    gpu.texture_count = static_cast<int>(scene.textures.size());
    packed.textures.resize(scene.textures.size());
    for (int i = 0; i < gpu.texture_count; ++i) {
        const std::shared_ptr<Texture>& texture = scene.textures[static_cast<size_t>(i)];
        if (!texture || texture->width <= 0 || texture->height <= 0) {
            return false;
        }
        const size_t pixels = static_cast<size_t>(texture->width) * static_cast<size_t>(texture->height);
        if (pixels != texture->pixels.size() || !size_fits_int(pixels)) {
            return false;
        }
        packed.textures[static_cast<size_t>(i)] = {texture->width, texture->height, 0};
    }
    gpu.material_count = static_cast<int>(scene.materials.size());
    packed.materials.resize(scene.materials.size());
    for (int i = 0; i < gpu.material_count; ++i) {
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(i)];
        if (!material) {
            return false;
        }
        float roughness = 0.5f;
        float metallic = 0.0f;
        if (const auto* principled = dynamic_cast<const PrincipledMaterial*>(material.get())) {
            roughness = principled->roughness;
            metallic = principled->metallic;
        } else if (const auto* dielectric = dynamic_cast<const DielectricMaterial*>(material.get())) {
            roughness = dielectric->ior;
        }
        int texture_index = -1;
        if (material->albedo_texture) {
            for (int t = 0; t < gpu.texture_count; ++t) {
                if (scene.textures[static_cast<size_t>(t)] == material->albedo_texture) {
                    texture_index = t;
                    break;
                }
            }
        }
        const float packed_roughness = material->model() == BrdfModel::Dielectric ? std::clamp(roughness, 1.0f, 3.0f) : std::clamp(roughness, 0.02f, 1.0f);
        packed.materials[static_cast<size_t>(i)] = {material->albedo, static_cast<int>(material->model()), packed_roughness, std::clamp(metallic, 0.0f, 1.0f), texture_index};
    }
    if (scene.environment.texture) {
        for (int t = 0; t < gpu.texture_count; ++t) {
            if (scene.textures[static_cast<size_t>(t)] == scene.environment.texture) {
                gpu.environment_texture = t;
                break;
            }
        }
        if (gpu.environment_texture < 0) {
            return false;
        }
    }
    const RenderScene render_scene = build_render_scene(scene);
    if (!size_fits_int(render_scene.triangles.size()) || !size_fits_int(render_scene.triangle_indices.size()) ||
        !size_fits_int(render_scene.flat_triangle_indices.size()) || !size_fits_int(render_scene.flat_bvh_nodes.size()) ||
        !size_fits_int(render_scene.bvh_nodes.size()) || !size_fits_int(render_scene.light_triangle_indices.size()) ||
        !size_fits_int(render_scene.mesh_instances.size()) || !size_fits_int(render_scene.mesh_instance_indices.size()) ||
        !size_fits_int(render_scene.tlas_nodes.size())) {
        return false;
    }
    gpu.triangle_count = static_cast<int>(render_scene.triangles.size());
    gpu.bvh_node_count = gpu.use_two_level ? static_cast<int>(render_scene.bvh_nodes.size()) : static_cast<int>(render_scene.flat_bvh_nodes.size());
    gpu.mesh_instance_count = static_cast<int>(render_scene.mesh_instances.size());
    gpu.tlas_node_count = static_cast<int>(render_scene.tlas_nodes.size());
    packed.triangles.resize(render_scene.triangles.size());
    packed.triangle_indices = gpu.use_two_level ? render_scene.triangle_indices : render_scene.flat_triangle_indices;
    const std::vector<BvhNode>& source_bvh_nodes = gpu.use_two_level ? render_scene.bvh_nodes : render_scene.flat_bvh_nodes;
    packed.bvh_nodes.resize(source_bvh_nodes.size());
    packed.mesh_instances.resize(render_scene.mesh_instances.size());
    packed.mesh_instance_indices = render_scene.mesh_instance_indices;
    packed.tlas_nodes.resize(render_scene.tlas_nodes.size());
    for (size_t i = 0; i < source_bvh_nodes.size(); ++i) {
        const BvhNode& node = source_bvh_nodes[i];
        packed.bvh_nodes[i] = {node.bounds.min, node.bounds.max, node.left, node.right, node.first, node.count};
    }
    for (size_t i = 0; i < render_scene.mesh_instances.size(); ++i) {
        const RenderScene::MeshInstance& instance = render_scene.mesh_instances[i];
        packed.mesh_instances[i] = {instance.bounds.min, instance.bounds.max, instance.bvh_root, instance.mesh};
    }
    for (size_t i = 0; i < render_scene.tlas_nodes.size(); ++i) {
        const BvhNode& node = render_scene.tlas_nodes[i];
        packed.tlas_nodes[i] = {node.bounds.min, node.bounds.max, node.left, node.right, node.first, node.count};
    }
    packed.light_indices.reserve(render_scene.light_triangle_indices.size());
    for (int i = 0; i < gpu.triangle_count; ++i) {
        const Triangle& tri = render_scene.triangles[static_cast<size_t>(i)];
        if (tri.material < 0 || tri.material >= gpu.material_count) {
            return false;
        }
        Vec3 emission;
        if (tri.mesh >= 0 && tri.mesh < static_cast<int>(scene.meshes.size())) {
            const LightComponent& light = scene.meshes[static_cast<size_t>(tri.mesh)].light;
            if (light.enabled && light.intensity > 0.0f) {
                emission = light.color * light.intensity;
            }
        }
        packed.triangles[static_cast<size_t>(i)] = {tri.v0, tri.v1, tri.v2, tri.normal, tri.n0, tri.n1, tri.n2, tri.centroid, emission, tri.uv0, tri.uv1, tri.uv2, tri.material, tri.mesh};
    }
    for (int i = 0; i < static_cast<int>(packed.triangle_indices.size()); ++i) {
        const int tri_index = packed.triangle_indices[static_cast<size_t>(i)];
        if (tri_index < 0 || tri_index >= gpu.triangle_count) {
            return false;
        }
        packed.triangle_indices[static_cast<size_t>(i)] = tri_index;
    }
    for (int i = 0; i < static_cast<int>(render_scene.light_triangle_indices.size()); ++i) {
        const int light_index = render_scene.light_triangle_indices[static_cast<size_t>(i)];
        if (light_index < 0 || light_index >= gpu.triangle_count) {
            return false;
        }
        packed.light_indices.push_back(light_index);
    }
    gpu.light_count = static_cast<int>(packed.light_indices.size());
    return true;
}

template <typename T>
bool upload_buffer(void*& device, int& cached_count, const std::vector<T>& host) {
    const int count = static_cast<int>(host.size());
    if (cached_count != count) {
        cudaFree(device);
        device = nullptr;
        cached_count = 0;
    }
    if (count == 0) {
        return true;
    }
    if (!device && cudaMalloc(&device, host.size() * sizeof(T)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    cached_count = count;
    return true;
}

void release_texture_objects(std::vector<void*>& arrays, std::vector<uint64_t>& objects) {
    for (uint64_t object : objects) {
        if (object != 0) {
            cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(object));
        }
    }
    for (void* array : arrays) {
        if (array) {
            cudaFreeArray(static_cast<cudaArray_t>(array));
        }
    }
    arrays.clear();
    objects.clear();
}

bool upload_texture_objects(const Scene& scene, PackedGpuScene& packed, std::vector<void*>& arrays, std::vector<uint64_t>& objects) {
    release_texture_objects(arrays, objects);
    arrays.resize(scene.textures.size(), nullptr);
    objects.resize(scene.textures.size(), 0);
    for (int i = 0; i < static_cast<int>(scene.textures.size()); ++i) {
        const std::shared_ptr<Texture>& texture = scene.textures[static_cast<size_t>(i)];
        if (!texture || texture->width <= 0 || texture->height <= 0) {
            return false;
        }
        std::vector<float4> pixels(texture->pixels.size());
        for (size_t p = 0; p < texture->pixels.size(); ++p) {
            const Vec3 color = texture->pixels[p];
            pixels[p] = make_float4(color.x, color.y, color.z, 1.0f);
        }

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaArray_t array = nullptr;
        if (cudaMallocArray(&array, &channel_desc, static_cast<size_t>(texture->width), static_cast<size_t>(texture->height)) != cudaSuccess) {
            release_texture_objects(arrays, objects);
            return false;
        }
        arrays[static_cast<size_t>(i)] = array;
        const size_t pitch = static_cast<size_t>(texture->width) * sizeof(float4);
        if (cudaMemcpy2DToArray(array, 0, 0, pixels.data(), pitch, pitch, static_cast<size_t>(texture->height), cudaMemcpyHostToDevice) != cudaSuccess) {
            release_texture_objects(arrays, objects);
            return false;
        }

        cudaResourceDesc resource_desc{};
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = array;
        cudaTextureDesc texture_desc{};
        texture_desc.addressMode[0] = cudaAddressModeWrap;
        texture_desc.addressMode[1] = cudaAddressModeWrap;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = 1;
        cudaTextureObject_t object = 0;
        if (cudaCreateTextureObject(&object, &resource_desc, &texture_desc, nullptr) != cudaSuccess) {
            release_texture_objects(arrays, objects);
            return false;
        }
        objects[static_cast<size_t>(i)] = static_cast<uint64_t>(object);
        packed.textures[static_cast<size_t>(i)].object = object;
    }
    return true;
}

bool apply_cached_texture_objects(PackedGpuScene& packed, const std::vector<uint64_t>& objects) {
    if (objects.size() != packed.textures.size()) {
        return false;
    }
    for (size_t i = 0; i < packed.textures.size(); ++i) {
        packed.textures[i].object = static_cast<cudaTextureObject_t>(objects[i]);
    }
    return true;
}

template <typename T>
bool upload_scene_field(GpuScene* device_scene, size_t offset, const T& value) {
    char* base = reinterpret_cast<char*>(device_scene);
    return cudaMemcpy(base + offset, &value, sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool upload_camera(GpuScene* device_scene, const Camera& camera) {
    return upload_scene_field(device_scene, offsetof(GpuScene, camera), camera);
}

bool upload_environment(GpuScene* device_scene, const GpuScene& scene) {
    return upload_scene_field(device_scene, offsetof(GpuScene, environment_texture), scene.environment_texture) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_color), scene.environment_color) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_strength), scene.environment_strength) &&
           upload_scene_field(device_scene, offsetof(GpuScene, environment_constant), scene.environment_constant);
}

bool make_environment_gpu(const Scene& scene, GpuScene& gpu) {
    gpu.environment_texture = -1;
    gpu.environment_color = scene.environment.color;
    gpu.environment_strength = scene.environment.strength;
    gpu.environment_constant = scene.environment.constant;
    if (!scene.environment.texture) {
        return true;
    }
    for (int i = 0; i < static_cast<int>(scene.textures.size()); ++i) {
        if (scene.textures[static_cast<size_t>(i)] == scene.environment.texture) {
            gpu.environment_texture = i;
            return true;
        }
    }
    return false;
}

} // namespace

bool CudaPathTracer::available() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaPathTracer::~CudaPathTracer() {
    reset();
}

void CudaPathTracer::reset() {
    cudaFree(device_accumulation_);
    cudaFree(device_rgba_);
    cudaFree(device_primary_hits_);
    cudaFree(device_scene_);
    cudaFree(device_materials_);
    cudaFree(device_textures_);
    cudaFree(device_triangles_);
    cudaFree(device_triangle_indices_);
    cudaFree(device_light_indices_);
    cudaFree(device_bvh_nodes_);
    cudaFree(device_mesh_instances_);
    cudaFree(device_mesh_instance_indices_);
    cudaFree(device_tlas_nodes_);
    device_accumulation_ = nullptr;
    device_rgba_ = nullptr;
    device_primary_hits_ = nullptr;
    device_scene_ = nullptr;
    device_materials_ = nullptr;
    device_textures_ = nullptr;
    device_triangles_ = nullptr;
    device_triangle_indices_ = nullptr;
    device_light_indices_ = nullptr;
    device_bvh_nodes_ = nullptr;
    device_mesh_instances_ = nullptr;
    device_mesh_instance_indices_ = nullptr;
    device_tlas_nodes_ = nullptr;
    cached_pixels_ = 0;
    cached_primary_pixels_ = 0;
    cached_materials_ = 0;
    cached_textures_ = 0;
    cached_triangles_ = 0;
    cached_triangle_indices_ = 0;
    cached_lights_ = 0;
    cached_bvh_nodes_ = 0;
    cached_mesh_instances_ = 0;
    cached_mesh_instance_indices_ = 0;
    cached_tlas_nodes_ = 0;
    scene_uploaded_ = false;
    primary_cache_valid_ = false;
    release_texture_objects(texture_arrays_, texture_objects_);
}

void CudaPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    const size_t pixels = static_cast<size_t>(settings.width) * static_cast<size_t>(settings.height);

    if (cached_pixels_ != pixels) {
        reset();
    }

    if (!device_accumulation_ && cudaMalloc(&device_accumulation_, pixels * sizeof(Vec3)) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    if (!device_rgba_ && cudaMalloc(&device_rgba_, pixels * sizeof(uint32_t)) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    if (settings.use_primary_hit_cache && cached_primary_pixels_ != pixels) {
        cudaFree(device_primary_hits_);
        device_primary_hits_ = nullptr;
        cached_primary_pixels_ = 0;
        primary_cache_valid_ = false;
    }
    if (settings.use_primary_hit_cache && !device_primary_hits_ && cudaMalloc(&device_primary_hits_, pixels * sizeof(GpuPrimaryHit)) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    if (settings.use_primary_hit_cache) {
        cached_primary_pixels_ = pixels;
    }
    if (!device_scene_ && cudaMalloc(&device_scene_, sizeof(GpuScene)) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    cached_pixels_ = pixels;

    Vec3* device_accumulation = static_cast<Vec3*>(device_accumulation_);
    uint32_t* device_rgba = static_cast<uint32_t*>(device_rgba_);
    GpuScene* device_scene = static_cast<GpuScene*>(device_scene_);

    const bool full_upload = !scene_uploaded_ ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture);
    if (!settings.use_primary_hit_cache ||
        has_dirty(settings.dirty, RenderDirty::Camera) ||
        has_dirty(settings.dirty, RenderDirty::Geometry)) {
        primary_cache_valid_ = false;
    }

    if (full_upload) {
        PackedGpuScene packed;
        if (!pack_scene(scene, settings, packed)) {
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        const bool upload_textures = texture_objects_.size() != packed.textures.size() || has_dirty(settings.dirty, RenderDirty::Texture);
        if (upload_textures ? !upload_texture_objects(scene, packed, texture_arrays_, texture_objects_) : !apply_cached_texture_objects(packed, texture_objects_)) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        if (!upload_buffer(device_materials_, cached_materials_, packed.materials) ||
            !upload_buffer(device_textures_, cached_textures_, packed.textures) ||
            !upload_buffer(device_triangles_, cached_triangles_, packed.triangles) ||
            !upload_buffer(device_triangle_indices_, cached_triangle_indices_, packed.triangle_indices) ||
            !upload_buffer(device_light_indices_, cached_lights_, packed.light_indices) ||
            !upload_buffer(device_bvh_nodes_, cached_bvh_nodes_, packed.bvh_nodes) ||
            !upload_buffer(device_mesh_instances_, cached_mesh_instances_, packed.mesh_instances) ||
            !upload_buffer(device_mesh_instance_indices_, cached_mesh_instance_indices_, packed.mesh_instance_indices) ||
            !upload_buffer(device_tlas_nodes_, cached_tlas_nodes_, packed.tlas_nodes)) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        packed.scene.materials = static_cast<GpuMaterial*>(device_materials_);
        packed.scene.textures = static_cast<GpuTexture*>(device_textures_);
        packed.scene.triangles = static_cast<GpuTriangle*>(device_triangles_);
        packed.scene.triangle_indices = static_cast<int*>(device_triangle_indices_);
        packed.scene.light_indices = static_cast<int*>(device_light_indices_);
        packed.scene.bvh_nodes = static_cast<GpuBvhNode*>(device_bvh_nodes_);
        packed.scene.mesh_instances = static_cast<GpuMeshInstance*>(device_mesh_instances_);
        packed.scene.mesh_instance_indices = static_cast<int*>(device_mesh_instance_indices_);
        packed.scene.tlas_nodes = static_cast<GpuBvhNode*>(device_tlas_nodes_);
        if (cudaMemcpy(device_scene, &packed.scene, sizeof(GpuScene), cudaMemcpyHostToDevice) != cudaSuccess) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        scene_uploaded_ = true;
    } else {
        if (has_dirty(settings.dirty, RenderDirty::Camera) && !upload_camera(device_scene, scene.camera)) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        if (has_dirty(settings.dirty, RenderDirty::Environment)) {
            GpuScene environment_scene;
            if (!make_environment_gpu(scene, environment_scene) || !upload_environment(device_scene, environment_scene)) {
                reset();
                CpuPathTracer fallback;
                fallback.render(scene, settings, framebuffer);
                return;
            }
        }
    }

    if (settings.frame_index == 0u || has_dirty(settings.dirty, RenderDirty::Render)) {
        if (cudaMemset(device_accumulation, 0, pixels * sizeof(Vec3)) != cudaSuccess) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
    }

    if (settings.frame_index == 0u) {
        std::fill(framebuffer.accumulation.begin(), framebuffer.accumulation.end(), Vec3{});
    }

    const dim3 block(16, 16);
    const dim3 grid((settings.width + block.x - 1) / block.x, (settings.height + block.y - 1) / block.y);
    render_kernel<<<grid, block>>>(device_scene, settings, device_accumulation, device_rgba, static_cast<GpuPrimaryHit*>(device_primary_hits_), primary_cache_valid_ ? 1 : 0);
    const cudaError_t render_error = cudaDeviceSynchronize();
    if (render_error != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    if (settings.use_primary_hit_cache) {
        primary_cache_valid_ = true;
    }

    const cudaError_t rgba_error = cudaMemcpy(framebuffer.rgba.data(), device_rgba, pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (rgba_error != cudaSuccess) {
        reset();
        framebuffer.clear();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
    }
}

} // namespace lt

#endif

#include "lt/renderer.h"

#if LT_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>

namespace lt {
namespace {

constexpr int kMaxMaterials = 64;
constexpr int kMaxTriangles = 4096;
constexpr int kMaxBvhNodes = kMaxTriangles * 2;
constexpr int kMaxTextures = 8;
constexpr int kMaxTexturePixels = 262144;

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
    int first_pixel = 0;
};

struct GpuTriangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
    Vec3 centroid;
    Vec3 emission;
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    int material = 0;
    int mesh = -1;
};

struct GpuScene {
    Camera camera;
    int material_count = 0;
    int triangle_count = 0;
    int bvh_node_count = 0;
    int light_count = 0;
    int texture_count = 0;
    int texture_pixel_count = 0;
    GpuMaterial materials[kMaxMaterials];
    GpuTexture textures[kMaxTextures];
    Vec3 texture_pixels[kMaxTexturePixels];
    GpuTriangle triangles[kMaxTriangles];
    int triangle_indices[kMaxTriangles];
    int light_indices[kMaxTriangles];
    BvhNode bvh_nodes[kMaxBvhNodes];
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
};

struct GpuMaterialSample {
    Vec3 direction;
    Vec3 weight;
    float pdf = 0.0f;
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

__device__ bool intersect_aabb_gpu(const Aabb& bounds, const Ray& ray, float max_t) {
    float tmin = 0.001f;
    float tmax = max_t;
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float direction = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float lo = axis == 0 ? bounds.min.x : axis == 1 ? bounds.min.y : bounds.min.z;
        const float hi = axis == 0 ? bounds.max.x : axis == 1 ? bounds.max.y : bounds.max.z;
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
    return true;
}

__device__ bool intersect_gpu(const GpuScene& scene, const Ray& ray, GpuHit& hit) {
    if (scene.bvh_node_count <= 0) {
        return false;
    }

    bool found = false;
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= scene.bvh_node_count) {
            continue;
        }
        const BvhNode node = scene.bvh_nodes[node_index];
        if (!intersect_aabb_gpu(node.bounds, ray, hit.t)) {
            continue;
        }

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= scene.triangle_count) {
                    continue;
                }
                const int tri_index = scene.triangle_indices[index_offset];
                if (tri_index < 0 || tri_index >= scene.triangle_count) {
                    continue;
                }
                const GpuTriangle tri = scene.triangles[tri_index];
                if (tri.material < 0 || tri.material >= scene.material_count) {
                    continue;
                }
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(tri, ray, t, u, v) && t < hit.t) {
                    hit.t = t;
                    hit.position = add(ray.origin, mul(ray.direction, t));
                    hit.normal = ddot(tri.normal, ray.direction) < 0.0f ? tri.normal : mul(tri.normal, -1.0f);
                    hit.uv = add(add(mul(tri.uv0, 1.0f - u - v), mul(tri.uv1, u)), mul(tri.uv2, v));
                    hit.material = tri.material;
                    hit.mesh = tri.mesh;
                    hit.triangle = tri_index;
                    hit.emission = tri.emission;
                    found = true;
                }
            }
        } else {
            if (node.left >= 0 && node.left < scene.bvh_node_count && stack_size < 64) {
                stack[stack_size++] = node.left;
            }
            if (node.right >= 0 && node.right < scene.bvh_node_count && stack_size < 64) {
                stack[stack_size++] = node.right;
            }
        }
    }
    return found;
}

__host__ __device__ bool has_light_emission_gpu(Vec3 emission) {
    return emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
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
    const float u = wrap01_gpu(uv.x);
    const float v = wrap01_gpu(uv.y);
    const int x = iclamp_gpu(static_cast<int>(u * static_cast<float>(texture.width)), 0, texture.width - 1);
    const int y = iclamp_gpu(static_cast<int>((1.0f - v) * static_cast<float>(texture.height)), 0, texture.height - 1);
    const int index = texture.first_pixel + y * texture.width + x;
    return index >= 0 && index < scene.texture_pixel_count ? scene.texture_pixels[index] : Vec3{1.0f, 1.0f, 1.0f};
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
    const float roughness = dclamp(material.roughness, 0.02f, 1.0f);
    const float metallic = dclamp(material.metallic, 0.0f, 1.0f);
    const float spec_prob = dclamp(0.25f + 0.75f * metallic, 0.25f, 1.0f);
    const Vec3 h = dnormalize(add(wi, wo));
    const float ndoth = fmaxf(0.0f, ddot(n, h));
    const float vdoth = fmaxf(0.0f, ddot(wo, h));
    const float specular_pdf = ggx_distribution_gpu(ndoth, roughness) * ndoth / fmaxf(1.0e-6f, 4.0f * vdoth);
    return (1.0f - spec_prob) * diffuse_pdf + spec_prob * specular_pdf;
}

__device__ GpuMaterialSample sample_material_gpu(const GpuScene& scene, const GpuMaterial& material, Vec3 n, Vec3 wo, Vec2 uv, uint32_t& rng) {
    GpuMaterialSample result;
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

__device__ Vec3 estimate_direct_gpu(const GpuScene& scene, const GpuHit& hit, const GpuMaterial& material, Vec3 wo, uint32_t& rng) {
    Vec3 direct{};
    for (int i = 0; i < scene.light_count; ++i) {
        const int light_index = scene.light_indices[i];
        if (light_index < 0 || light_index >= scene.triangle_count) {
            continue;
        }
        const GpuTriangle light = scene.triangles[light_index];
        if (!has_light_emission_gpu(light.emission)) {
            continue;
        }
        const Vec3 light_point = sample_triangle_area_gpu(light, rng);
        const Vec3 to_light = sub(light_point, hit.position);
        const float dist2 = ddot(to_light, to_light);
        if (dist2 <= 1.0e-8f) {
            continue;
        }
        const float dist = sqrtf(dist2);
        const Vec3 light_dir = divv(to_light, dist);
        const float ndotl = fmaxf(0.0f, ddot(hit.normal, light_dir));
        const float ldot = fmaxf(0.0f, ddot(mul(light.normal, -1.0f), light_dir));
        if (ndotl <= 0.0f || ldot <= 0.0f) {
            continue;
        }
        GpuHit shadow_hit;
        const Ray shadow_ray{add(hit.position, mul(hit.normal, 0.002f)), light_dir};
        if (intersect_gpu(scene, shadow_ray, shadow_hit)) {
            if (shadow_hit.t < dist - 0.01f || shadow_hit.triangle != light_index) {
                continue;
            }
        }
        const float area = 0.5f * sqrtf(ddot(dcross(sub(light.v1, light.v0), sub(light.v2, light.v0)), dcross(sub(light.v1, light.v0), sub(light.v2, light.v0))));
        direct = add(direct, mul(mul(evaluate_brdf_gpu(scene, material, hit.normal, wo, light_dir, hit.uv), light.emission), ndotl * ldot * area / fmaxf(0.001f, dist2)));
    }
    return direct;
}

__device__ Vec3 trace_gpu(const GpuScene& scene, Ray ray, uint32_t& rng, int max_bounces) {
    Vec3 radiance{};
    Vec3 throughput{1.0f, 1.0f, 1.0f};
    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        GpuHit hit;
        if (!intersect_gpu(scene, ray, hit)) {
            const float t = 0.5f * (ray.direction.y + 1.0f);
            radiance = add(radiance, mul(throughput, add(mul(Vec3{0.02f, 0.025f, 0.035f}, 1.0f - t), mul(Vec3{0.32f, 0.45f, 0.68f}, t))));
            break;
        }
        if (hit.material < 0 || hit.material >= scene.material_count) {
            break;
        }
        const GpuMaterial material = scene.materials[hit.material];
        if (has_light_emission_gpu(hit.emission)) {
            if (bounce == 0) {
                radiance = add(radiance, mul(throughput, hit.emission));
            }
            break;
        }
        const Vec3 wo = mul(ray.direction, -1.0f);
        radiance = add(radiance, mul(throughput, estimate_direct_gpu(scene, hit, material, wo, rng)));
        if (bounce >= 3) {
            const float p = dclamp(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng_float(rng) > p) {
                break;
            }
            throughput = divv(throughput, p);
        }
        const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal, wo, hit.uv, rng);
        if (sample.pdf <= 0.0f || ddot(sample.weight, sample.weight) <= 0.0f) {
            break;
        }
        ray = {add(hit.position, mul(hit.normal, 0.001f)), sample.direction};
        throughput = mul(throughput, sample.weight);
    }
    return radiance;
}

__global__ void render_kernel(const GpuScene* scene_ptr, RenderSettings settings, Vec3* accumulation, uint32_t* rgba) {
    const GpuScene& scene = *scene_ptr;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) {
        return;
    }
    const int idx = y * settings.width + x;
    uint32_t rng = make_pixel_seed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), settings.frame_index);
    Vec3 sample{};
    for (int s = 0; s < settings.samples_per_pixel; ++s) {
        sample = add(sample, trace_gpu(scene, camera_ray(scene.camera, x, y, settings.width, settings.height, rng), rng, settings.max_bounces));
    }
    sample = divv(sample, static_cast<float>(settings.samples_per_pixel));
    accumulation[idx] = add(accumulation[idx], sample);
    rgba[idx] = rgba8_gpu(divv(accumulation[idx], static_cast<float>(settings.frame_index + 1u)));
}

bool pack_scene(const Scene& scene, GpuScene& gpu) {
    std::memset(&gpu, 0, sizeof(GpuScene));
    gpu.camera = scene.camera;
    gpu.texture_count = std::min(static_cast<int>(scene.textures.size()), kMaxTextures);
    for (int i = 0; i < gpu.texture_count; ++i) {
        const std::shared_ptr<Texture>& texture = scene.textures[static_cast<size_t>(i)];
        if (!texture || texture->width <= 0 || texture->height <= 0) {
            return false;
        }
        const int pixels = texture->width * texture->height;
        if (pixels < 0 || gpu.texture_pixel_count + pixels > kMaxTexturePixels) {
            return false;
        }
        gpu.textures[i] = {texture->width, texture->height, gpu.texture_pixel_count};
        for (int p = 0; p < pixels; ++p) {
            gpu.texture_pixels[gpu.texture_pixel_count++] = texture->pixels[static_cast<size_t>(p)];
        }
    }
    gpu.material_count = std::min(static_cast<int>(scene.materials.size()), kMaxMaterials);
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
        gpu.materials[i] = {material->albedo, static_cast<int>(material->model()), std::clamp(roughness, 0.02f, 1.0f), std::clamp(metallic, 0.0f, 1.0f), texture_index};
    }
    const RenderScene render_scene = build_render_scene(scene);
    if (render_scene.triangles.size() > kMaxTriangles || render_scene.bvh_nodes.size() > kMaxBvhNodes) {
        return false;
    }
    gpu.triangle_count = std::min(static_cast<int>(render_scene.triangles.size()), kMaxTriangles);
    gpu.bvh_node_count = std::min(static_cast<int>(render_scene.bvh_nodes.size()), kMaxBvhNodes);
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
        gpu.triangles[i] = {tri.v0, tri.v1, tri.v2, tri.normal, tri.centroid, emission, tri.uv0, tri.uv1, tri.uv2, tri.material, tri.mesh};
    }
    for (int i = 0; i < gpu.triangle_count; ++i) {
        const int tri_index = render_scene.triangle_indices[static_cast<size_t>(i)];
        if (tri_index < 0 || tri_index >= gpu.triangle_count) {
            return false;
        }
        gpu.triangle_indices[i] = tri_index;
    }
    for (int i = 0; i < gpu.bvh_node_count; ++i) {
        gpu.bvh_nodes[i] = render_scene.bvh_nodes[static_cast<size_t>(i)];
    }
    if (render_scene.light_triangle_indices.size() > kMaxTriangles) {
        return false;
    }
    for (int i = 0; i < static_cast<int>(render_scene.light_triangle_indices.size()); ++i) {
        const int light_index = render_scene.light_triangle_indices[static_cast<size_t>(i)];
        if (light_index < 0 || light_index >= gpu.triangle_count) {
            return false;
        }
        gpu.light_indices[gpu.light_count++] = light_index;
    }
    return true;
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
    cudaFree(device_scene_);
    device_accumulation_ = nullptr;
    device_rgba_ = nullptr;
    device_scene_ = nullptr;
    cached_pixels_ = 0;
}

void CudaPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    const size_t pixels = static_cast<size_t>(settings.width) * static_cast<size_t>(settings.height);
    const std::unique_ptr<GpuScene> gpu_scene = std::make_unique<GpuScene>();
    if (!pack_scene(scene, *gpu_scene)) {
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }

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

    if (cudaMemcpy(device_accumulation, framebuffer.accumulation.data(), pixels * sizeof(Vec3), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_scene, gpu_scene.get(), sizeof(GpuScene), cudaMemcpyHostToDevice) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }

    const dim3 block(16, 16);
    const dim3 grid((settings.width + block.x - 1) / block.x, (settings.height + block.y - 1) / block.y);
    render_kernel<<<grid, block>>>(device_scene, settings, device_accumulation, device_rgba);
    const cudaError_t render_error = cudaDeviceSynchronize();
    if (render_error != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }

    const cudaError_t accum_error = cudaMemcpy(framebuffer.accumulation.data(), device_accumulation, pixels * sizeof(Vec3), cudaMemcpyDeviceToHost);
    const cudaError_t rgba_error = cudaMemcpy(framebuffer.rgba.data(), device_rgba, pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (accum_error != cudaSuccess || rgba_error != cudaSuccess) {
        reset();
        framebuffer.clear();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
    }
}

} // namespace lt

#endif

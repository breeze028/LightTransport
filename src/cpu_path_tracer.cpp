#include "lt/renderer.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

namespace lt {
namespace {

constexpr int kTraversalStackSize = 64;

struct Hit {
    float t = kInfinity;
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    int material = -1;
    int mesh = -1;
    int triangle = -1;
};

Ray make_camera_ray(const Camera& camera, int x, int y, const RenderSettings& settings, Rng& rng) {
    const float aspect = static_cast<float>(settings.width) / static_cast<float>(settings.height);
    const float half_height = std::tan(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = normalize(camera.target - camera.position);
    const Vec3 right = normalize(cross(forward, camera.up));
    const Vec3 up = cross(right, forward);
    const float u = ((static_cast<float>(x) + rng.next_float()) / static_cast<float>(settings.width) * 2.0f - 1.0f) * half_width;
    const float v = (1.0f - (static_cast<float>(y) + rng.next_float()) / static_cast<float>(settings.height) * 2.0f) * half_height;
    return {camera.position, normalize(forward + right * u + up * v)};
}

bool intersect_aabb(const Aabb& bounds, const Ray& ray, float max_t) {
    float tmin = 0.001f;
    float tmax = max_t;
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float direction = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float lo = axis == 0 ? bounds.min.x : axis == 1 ? bounds.min.y : bounds.min.z;
        const float hi = axis == 0 ? bounds.max.x : axis == 1 ? bounds.max.y : bounds.max.z;
        if (std::fabs(direction) < 1.0e-8f) {
            if (origin < lo || origin > hi) {
                return false;
            }
            continue;
        }
        const float inv = 1.0f / direction;
        float t0 = (lo - origin) * inv;
        float t1 = (hi - origin) * inv;
        if (inv < 0.0f) {
            std::swap(t0, t1);
        }
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        if (tmax < tmin) {
            return false;
        }
    }
    return true;
}

bool intersect_triangle(const Triangle& tri, const Ray& ray, float& t, float& u, float& v) {
    const Vec3 edge1 = tri.v1 - tri.v0;
    const Vec3 edge2 = tri.v2 - tri.v0;
    const Vec3 p = cross(ray.direction, edge2);
    const float det = dot(edge1, p);
    if (std::fabs(det) < 1.0e-7f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 s = ray.origin - tri.v0;
    u = inv_det * dot(s, p);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3 q = cross(s, edge1);
    v = inv_det * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    t = inv_det * dot(edge2, q);
    return t > 0.001f;
}

bool intersect_scene(const RenderScene& render_scene, const Ray& ray, Hit& hit) {
    if (render_scene.bvh_nodes.empty()) {
        return false;
    }

    bool found = false;
    int stack[kTraversalStackSize] = {};
    int stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        const BvhNode& node = render_scene.bvh_nodes[static_cast<size_t>(stack[--stack_size])];
        if (!intersect_aabb(node.bounds, ray, hit.t)) {
            continue;
        }
        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= static_cast<int>(render_scene.triangle_indices.size())) {
                    continue;
                }
                const int tri_index = render_scene.triangle_indices[static_cast<size_t>(index_offset)];
                if (tri_index < 0 || tri_index >= static_cast<int>(render_scene.triangles.size())) {
                    continue;
                }
                const Triangle& tri = render_scene.triangles[static_cast<size_t>(tri_index)];
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(tri, ray, t, u, v) && t < hit.t) {
                    hit.t = t;
                    hit.position = ray.origin + ray.direction * t;
                    hit.normal = face_forward(tri.normal, ray.direction);
                    hit.uv = tri.uv0 * (1.0f - u - v) + tri.uv1 * u + tri.uv2 * v;
                    hit.material = tri.material;
                    hit.mesh = tri.mesh;
                    hit.triangle = tri_index;
                    found = true;
                }
            }
        } else {
            if (node.left >= 0 && node.left < static_cast<int>(render_scene.bvh_nodes.size()) && stack_size < kTraversalStackSize) {
                stack[stack_size++] = node.left;
            }
            if (node.right >= 0 && node.right < static_cast<int>(render_scene.bvh_nodes.size()) && stack_size < kTraversalStackSize) {
                stack[stack_size++] = node.right;
            }
        }
    }
    return found;
}

Vec3 light_emission(const Scene& scene, int mesh_index) {
    if (mesh_index < 0 || mesh_index >= static_cast<int>(scene.meshes.size())) {
        return {};
    }
    const LightComponent& light = scene.meshes[static_cast<size_t>(mesh_index)].light;
    return light.enabled && light.intensity > 0.0f ? light.color * light.intensity : Vec3{};
}

bool has_light_emission(Vec3 emission) {
    return emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
}

Vec3 sample_triangle_area(const Triangle& tri, Rng& rng) {
    const float su = std::sqrt(rng.next_float());
    const float v = rng.next_float();
    return tri.v0 * (1.0f - su) + tri.v1 * (su * (1.0f - v)) + tri.v2 * (su * v);
}

uint64_t hash_combine(uint64_t h, uint64_t v) {
    return h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u));
}

uint64_t hash_float(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint64_t>(bits);
}

uint64_t hash_vec3(uint64_t h, Vec3 v) {
    h = hash_combine(h, hash_float(v.x));
    h = hash_combine(h, hash_float(v.y));
    h = hash_combine(h, hash_float(v.z));
    return h;
}

uint64_t scene_geometry_signature(const Scene& scene) {
    uint64_t h = 1469598103934665603ull;
    h = hash_combine(h, static_cast<uint64_t>(scene.meshes.size()));
    for (const Mesh& mesh : scene.meshes) {
        h = hash_combine(h, static_cast<uint64_t>(mesh.material));
        h = hash_vec3(h, mesh.translation);
        h = hash_vec3(h, mesh.rotation);
        h = hash_vec3(h, mesh.scale);
        h = hash_combine(h, static_cast<uint64_t>(mesh.light.enabled));
        h = hash_vec3(h, mesh.light.color);
        h = hash_combine(h, hash_float(mesh.light.intensity));
        h = hash_combine(h, static_cast<uint64_t>(mesh.vertices.size()));
        h = hash_combine(h, static_cast<uint64_t>(mesh.texcoords.size()));
        h = hash_combine(h, static_cast<uint64_t>(mesh.indices.size()));
        for (Vec3 vertex : mesh.vertices) {
            h = hash_vec3(h, vertex);
        }
        for (Vec2 uv : mesh.texcoords) {
            h = hash_combine(h, hash_float(uv.x));
            h = hash_combine(h, hash_float(uv.y));
        }
        for (uint32_t index : mesh.indices) {
            h = hash_combine(h, static_cast<uint64_t>(index));
        }
    }
    return h;
}

Vec3 estimate_direct_lighting(const RenderScene& render_scene, const Scene& scene, const Hit& hit, const Material& material, Vec3 wo, Rng& rng) {
    Vec3 direct;
    for (int light_index : render_scene.light_triangle_indices) {
        if (light_index < 0 || light_index >= static_cast<int>(render_scene.triangles.size())) {
            continue;
        }
        const Triangle& light = render_scene.triangles[static_cast<size_t>(light_index)];
        const Vec3 emission = light_emission(scene, light.mesh);
        if (!has_light_emission(emission)) {
            continue;
        }

        const Vec3 light_point = sample_triangle_area(light, rng);
        const Vec3 to_light = light_point - hit.position;
        const float dist2 = dot(to_light, to_light);
        if (dist2 <= 1.0e-8f) {
            continue;
        }
        const float dist = std::sqrt(dist2);
        const Vec3 light_dir = to_light / dist;
        const float ndotl = std::max(0.0f, dot(hit.normal, light_dir));
        const float ldot = std::max(0.0f, dot(-light.normal, light_dir));
        if (ndotl <= 0.0f || ldot <= 0.0f) {
            continue;
        }

        Hit shadow_hit;
        const Ray shadow_ray{hit.position + hit.normal * 0.002f, light_dir};
        if (intersect_scene(render_scene, shadow_ray, shadow_hit)) {
            if (shadow_hit.t < dist - 0.01f || shadow_hit.triangle != light_index) {
                continue;
            }
        }

        const float area = 0.5f * length(cross(light.v1 - light.v0, light.v2 - light.v0));
        direct += material.evaluate(hit.normal, wo, light_dir, hit.uv) * emission * (ndotl * ldot * area / std::max(0.001f, dist2));
    }
    return direct;
}

Vec3 trace_path(const RenderScene& render_scene, const Scene& scene, Ray ray, Rng& rng, int max_bounces) {
    Vec3 radiance;
    Vec3 throughput{1.0f};

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        Hit hit;
        if (!intersect_scene(render_scene, ray, hit)) {
            const float t = 0.5f * (ray.direction.y + 1.0f);
            radiance += throughput * ((1.0f - t) * Vec3{0.02f, 0.025f, 0.035f} + t * Vec3{0.32f, 0.45f, 0.68f});
            break;
        }

        if (hit.material < 0 || hit.material >= static_cast<int>(scene.materials.size())) {
            break;
        }
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(hit.material)];
        if (!material) {
            break;
        }
        const Vec3 emission = light_emission(scene, hit.mesh);
        if (has_light_emission(emission)) {
            if (bounce == 0) {
                radiance += throughput * emission;
            }
            break;
        }
        const Vec3 wo = -ray.direction;
        radiance += throughput * estimate_direct_lighting(render_scene, scene, hit, *material, wo, rng);

        if (bounce >= 3) {
            const float p = std::clamp(std::max({throughput.x, throughput.y, throughput.z}), 0.05f, 0.95f);
            if (rng.next_float() > p) {
                break;
            }
            throughput = throughput / p;
        }

        const MaterialSample sample = material->sample(hit.normal, wo, hit.uv, rng);
        if (sample.pdf <= 0.0f || dot(sample.weight, sample.weight) <= 0.0f) {
            break;
        }
        ray = {hit.position + hit.normal * 0.001f, sample.direction};
        throughput = throughput * sample.weight;
    }
    return radiance;
}

} // namespace

void CpuPathTracer::reset() {
    cached_render_scene_ = {};
    cached_scene_signature_ = 0;
}

void CpuPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    const uint64_t scene_signature = scene_geometry_signature(scene);
    if (cached_scene_signature_ != scene_signature) {
        cached_render_scene_ = build_render_scene(scene);
        cached_scene_signature_ = scene_signature;
    }
    const RenderScene& render_scene = cached_render_scene_;

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const unsigned int thread_count = std::max(1u, hardware_threads > 1u ? hardware_threads - 1u : hardware_threads);
    std::atomic<int> next_row{0};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    const auto render_rows = [&]() {
        for (;;) {
            const int y = next_row.fetch_add(1, std::memory_order_relaxed);
            if (y >= settings.height) {
                break;
            }
            for (int x = 0; x < settings.width; ++x) {
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(settings.width) + static_cast<size_t>(x);
                Vec3 sample;
                Rng rng(make_pixel_seed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), settings.frame_index));
                for (int s = 0; s < settings.samples_per_pixel; ++s) {
                    sample += trace_path(render_scene, scene, make_camera_ray(scene.camera, x, y, settings, rng), rng, settings.max_bounces);
                }
                sample = sample / static_cast<float>(settings.samples_per_pixel);
                framebuffer.accumulation[idx] += sample;
                framebuffer.rgba[idx] = to_rgba8(framebuffer.accumulation[idx] / static_cast<float>(settings.frame_index + 1u));
            }
        }
    };

    for (unsigned int i = 0; i < thread_count; ++i) {
        workers.emplace_back(render_rows);
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

#if !LT_HAS_CUDA
CudaPathTracer::~CudaPathTracer() = default;
bool CudaPathTracer::available() const { return false; }
void CudaPathTracer::reset() {}
void CudaPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    CpuPathTracer fallback;
    fallback.render(scene, settings, framebuffer);
}
#endif

} // namespace lt

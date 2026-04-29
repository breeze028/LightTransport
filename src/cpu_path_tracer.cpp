#include "lt/renderer.h"

#include <algorithm>
#include <atomic>
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
    bool front_face = true;
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

bool intersect_bvh_nodes(const RenderScene& render_scene, const std::vector<BvhNode>& nodes, const std::vector<int>& indices, const Ray& ray, int root, Hit& hit) {
    bool found = false;
    int stack[kTraversalStackSize] = {};
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        const BvhNode& node = nodes[static_cast<size_t>(stack[--stack_size])];
        if (!intersect_aabb(node.bounds, ray, hit.t)) {
            continue;
        }
        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= static_cast<int>(indices.size())) {
                    continue;
                }
                const int tri_index = indices[static_cast<size_t>(index_offset)];
                if (tri_index < 0 || tri_index >= static_cast<int>(render_scene.triangles.size())) {
                    continue;
                }
                const Triangle& tri = render_scene.triangles[static_cast<size_t>(tri_index)];
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(tri, ray, t, u, v) && t < hit.t) {
                    const Vec3 shading_normal = normalize(tri.n0 * (1.0f - u - v) + tri.n1 * u + tri.n2 * v);
                    hit.t = t;
                    hit.position = ray.origin + ray.direction * t;
                    hit.front_face = dot(tri.normal, ray.direction) < 0.0f;
                    hit.normal = face_forward(dot(shading_normal, shading_normal) > 0.0f ? shading_normal : tri.normal, ray.direction);
                    hit.uv = tri.uv0 * (1.0f - u - v) + tri.uv1 * u + tri.uv2 * v;
                    hit.material = tri.material;
                    hit.mesh = tri.mesh;
                    hit.triangle = tri_index;
                    found = true;
                }
            }
        } else {
            if (node.left >= 0 && node.left < static_cast<int>(nodes.size()) && stack_size < kTraversalStackSize) {
                stack[stack_size++] = node.left;
            }
            if (node.right >= 0 && node.right < static_cast<int>(nodes.size()) && stack_size < kTraversalStackSize) {
                stack[stack_size++] = node.right;
            }
        }
    }
    return found;
}

bool intersect_blas(const RenderScene& render_scene, const Ray& ray, int root, Hit& hit) {
    return intersect_bvh_nodes(render_scene, render_scene.bvh_nodes, render_scene.triangle_indices, ray, root, hit);
}

bool use_two_level(AccelerationStructure acceleration_structure) {
    return acceleration_structure == AccelerationStructure::TwoLevel;
}

bool intersect_scene(const RenderScene& render_scene, const Ray& ray, Hit& hit, AccelerationStructure acceleration_structure) {
    if (!use_two_level(acceleration_structure)) {
        return !render_scene.flat_bvh_nodes.empty() &&
            intersect_bvh_nodes(render_scene, render_scene.flat_bvh_nodes, render_scene.flat_triangle_indices, ray, 0, hit);
    }
    if (render_scene.tlas_nodes.empty() || render_scene.mesh_instances.empty()) {
        return false;
    }

    bool found = false;
    int stack[kTraversalStackSize] = {};
    int stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        const BvhNode& node = render_scene.tlas_nodes[static_cast<size_t>(stack[--stack_size])];
        if (!intersect_aabb(node.bounds, ray, hit.t)) {
            continue;
        }
        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= static_cast<int>(render_scene.mesh_instance_indices.size())) {
                    continue;
                }
                const int instance_index = render_scene.mesh_instance_indices[static_cast<size_t>(index_offset)];
                if (instance_index < 0 || instance_index >= static_cast<int>(render_scene.mesh_instances.size())) {
                    continue;
                }
                const RenderScene::MeshInstance& instance = render_scene.mesh_instances[static_cast<size_t>(instance_index)];
                if (instance.bvh_root >= 0 && intersect_aabb(instance.bounds, ray, hit.t)) {
                    found = intersect_blas(render_scene, ray, instance.bvh_root, hit) || found;
                }
            }
        } else {
            if (node.left >= 0 && node.left < static_cast<int>(render_scene.tlas_nodes.size()) && stack_size < kTraversalStackSize) {
                stack[stack_size++] = node.left;
            }
            if (node.right >= 0 && node.right < static_cast<int>(render_scene.tlas_nodes.size()) && stack_size < kTraversalStackSize) {
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

Vec3 emitted_radiance(const Scene& scene, const Triangle& light, Vec3 ray_direction_to_light) {
    const Vec3 emission = light_emission(scene, light.mesh);
    return has_light_emission(emission) && dot(light.normal, -ray_direction_to_light) > 0.0f ? emission : Vec3{};
}

Vec3 environment_radiance(const Scene& scene, Vec3 direction) {
    if (scene.environment.texture) {
        const float u = std::atan2(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
        const float v = std::acos(std::clamp(direction.y, -1.0f, 1.0f)) / kPi;
        return scene.environment.color * scene.environment.texture->sample({u, 1.0f - v}) * scene.environment.strength;
    }
    if (scene.environment.constant) {
        return scene.environment.color * scene.environment.strength;
    }
    const float t = 0.5f * (direction.y + 1.0f);
    return ((1.0f - t) * Vec3{0.02f, 0.025f, 0.035f} + t * Vec3{0.32f, 0.45f, 0.68f}) * scene.environment.color * scene.environment.strength;
}

Vec3 sample_triangle_area(const Triangle& tri, Rng& rng) {
    const float su = std::sqrt(rng.next_float());
    const float v = rng.next_float();
    return tri.v0 * (1.0f - su) + tri.v1 * (su * (1.0f - v)) + tri.v2 * (su * v);
}

float triangle_area(const Triangle& tri) {
    return 0.5f * length(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

float light_pdf_solid_angle(const Triangle& light, Vec3 origin, Vec3 light_point, float light_pmf) {
    const Vec3 to_light = light_point - origin;
    const float dist2 = dot(to_light, to_light);
    if (dist2 <= 1.0e-8f) {
        return 0.0f;
    }
    const Vec3 light_dir = to_light / std::sqrt(dist2);
    const float ldot = std::max(0.0f, dot(-light.normal, light_dir));
    const float area = triangle_area(light);
    return ldot > 0.0f && area > 0.0f && light_pmf > 0.0f ? light_pmf * dist2 / (ldot * area) : 0.0f;
}

float mis_weight(float pdf_a, float pdf_b, MisHeuristic heuristic) {
    if (heuristic == MisHeuristic::Balance) {
        return pdf_a / std::max(1.0e-8f, pdf_a + pdf_b);
    }
    const float a = pdf_a;
    const float b = pdf_b;
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 / std::max(1.0e-8f, a2 + b2);
}

Vec3 estimate_direct_lighting(const RenderScene& render_scene, const Scene& scene, const Hit& hit, const Material& material, Vec3 wo, Rng& rng, const RenderSettings& settings) {
    Vec3 direct;
    const int light_count = static_cast<int>(render_scene.light_triangle_indices.size());
    if (light_count <= 0) {
        return direct;
    }
    const int light_list_index = std::min(static_cast<int>(rng.next_float() * static_cast<float>(light_count)), light_count - 1);
    const int light_index = render_scene.light_triangle_indices[static_cast<size_t>(light_list_index)];
    if (light_index < 0 || light_index >= static_cast<int>(render_scene.triangles.size())) {
        return direct;
    }
    const Triangle& light = render_scene.triangles[static_cast<size_t>(light_index)];
    const Vec3 emission = light_emission(scene, light.mesh);
    if (!has_light_emission(emission)) {
        return direct;
    }

    const Vec3 light_point = sample_triangle_area(light, rng);
    const Vec3 to_light = light_point - hit.position;
    const float dist2 = dot(to_light, to_light);
    if (dist2 <= 1.0e-8f) {
        return direct;
    }
    const float dist = std::sqrt(dist2);
    const Vec3 light_dir = to_light / dist;
    const float ndotl = std::max(0.0f, dot(hit.normal, light_dir));
    const float ldot = std::max(0.0f, dot(-light.normal, light_dir));
    if (ndotl <= 0.0f || ldot <= 0.0f) {
        return direct;
    }

    Hit shadow_hit;
    const Ray shadow_ray{hit.position + hit.normal * 0.002f, light_dir};
    if (intersect_scene(render_scene, shadow_ray, shadow_hit, settings.acceleration_structure)) {
        if (shadow_hit.t < dist - 0.01f || shadow_hit.triangle != light_index) {
            return direct;
        }
    }

    const float light_pmf = 1.0f / static_cast<float>(light_count);
    const float light_pdf = light_pdf_solid_angle(light, hit.position, light_point, light_pmf);
    const float bsdf_pdf = material.pdf(hit.normal, wo, light_dir);
    const float weight = settings.use_mis ? mis_weight(light_pdf, bsdf_pdf, settings.mis_heuristic) : 1.0f;
    direct += material.evaluate(hit.normal, wo, light_dir, hit.uv) * emission * (ndotl / std::max(1.0e-8f, light_pdf)) * weight;
    return direct;
}

Vec3 trace_path(const RenderScene& render_scene, const Scene& scene, Ray ray, Rng& rng, const RenderSettings& settings) {
    Vec3 radiance;
    Vec3 throughput{1.0f};
    Vec3 previous_position;
    float previous_bsdf_pdf = 0.0f;
    bool previous_delta = false;

    for (int bounce = 0; bounce < settings.max_bounces; ++bounce) {
        Hit hit;
        if (!intersect_scene(render_scene, ray, hit, settings.acceleration_structure)) {
            radiance += throughput * environment_radiance(scene, ray.direction);
            break;
        }

        if (hit.material < 0 || hit.material >= static_cast<int>(scene.materials.size())) {
            break;
        }
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(hit.material)];
        if (!material) {
            break;
        }
        Vec3 emission;
        if (hit.triangle >= 0 && hit.triangle < static_cast<int>(render_scene.triangles.size())) {
            emission = emitted_radiance(scene, render_scene.triangles[static_cast<size_t>(hit.triangle)], ray.direction);
        }
        if (has_light_emission(light_emission(scene, hit.mesh))) {
            if (bounce == 0) {
                radiance += throughput * emission;
            } else if (previous_delta) {
                radiance += throughput * emission;
            } else if (settings.use_mis && hit.triangle >= 0 && hit.triangle < static_cast<int>(render_scene.triangles.size())) {
                const Triangle& light = render_scene.triangles[static_cast<size_t>(hit.triangle)];
                const float light_pmf = render_scene.light_triangle_indices.empty() ? 0.0f : 1.0f / static_cast<float>(render_scene.light_triangle_indices.size());
                const float light_pdf = light_pdf_solid_angle(light, previous_position, hit.position, light_pmf);
                radiance += throughput * emission * mis_weight(previous_bsdf_pdf, light_pdf, settings.mis_heuristic);
            }
            break;
        }
        const Vec3 wo = -ray.direction;
        radiance += throughput * estimate_direct_lighting(render_scene, scene, hit, *material, wo, rng, settings);

        if (bounce >= 3) {
            const float p = std::clamp(std::max({throughput.x, throughput.y, throughput.z}), 0.05f, 0.95f);
            if (rng.next_float() > p) {
                break;
            }
            throughput = throughput / p;
        }

        const MaterialSample sample = material->sample(hit.normal, wo, hit.uv, hit.front_face, rng);
        if (sample.pdf <= 0.0f || dot(sample.weight, sample.weight) <= 0.0f) {
            break;
        }
        previous_position = hit.position;
        previous_bsdf_pdf = sample.pdf;
        previous_delta = sample.delta;
        const float offset_side = dot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
        ray = {hit.position + hit.normal * (0.001f * offset_side), sample.direction};
        throughput = throughput * sample.weight;
    }
    return radiance;
}

} // namespace

void CpuPathTracer::reset() {
    cached_render_scene_ = {};
    scene_uploaded_ = false;
}

void CpuPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    if (!scene_uploaded_ || has_dirty(settings.dirty, RenderDirty::Geometry)) {
        cached_render_scene_ = build_render_scene(scene);
        scene_uploaded_ = true;
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
                    sample += trace_path(render_scene, scene, make_camera_ray(scene.camera, x, y, settings, rng), rng, settings);
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

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
        sample = add(sample, trace_gpu_with_irradiance_probe_debug(scene, camera_ray(scene.camera, x, y, settings.width, settings.height, rng), rng, settings));
    }
    sample = divv(sample, static_cast<float>(settings.samples_per_pixel));
    accumulation[idx] = add(accumulation[idx], sample);
    rgba[idx] = rgba8_gpu(divv(accumulation[idx], static_cast<float>(settings.frame_index + 1u)));
}

struct GpuWavefrontPaths {
    Ray* rays = nullptr;
    Vec3* throughputs = nullptr;
    Vec3* radiance = nullptr;
    Vec3* previous_positions = nullptr;
    float* previous_bsdf_pdfs = nullptr;
    uint32_t* rngs = nullptr;
    int* pixels = nullptr;
    uint32_t* states = nullptr;
};

struct GpuWavefrontPathRef {
    Ray& ray;
    Vec3& throughput;
    Vec3& radiance;
    Vec3& previous_position;
    float& previous_bsdf_pdf;
    uint32_t& rng;
    int& pixel;
    uint32_t& state;
};

struct GpuWavefrontIntersectPaths {
    Ray* rays = nullptr;
    Vec3* throughputs = nullptr;
    Vec3* radiance = nullptr;
    uint32_t* rngs = nullptr;
    int* pixels = nullptr;
    uint32_t* states = nullptr;
};

struct GpuWavefrontIntersectPathRef {
    Ray& ray;
    Vec3& throughput;
    Vec3& radiance;
    uint32_t& rng;
    int& pixel;
    uint32_t& state;
};

struct GpuWavefrontSvgfAov {
    Vec3* albedo = nullptr;
    Vec3* emission = nullptr;
    Vec3* normal = nullptr;
    Vec3* world_position = nullptr;
    float* depth = nullptr;
    uint32_t* object_id = nullptr;
};

enum GpuWavefrontQueue : int {
    GpuWavefrontQueueActive = 0,
    GpuWavefrontQueueDirectLight = 1,
    GpuWavefrontQueueGi = 2,
    GpuWavefrontQueueShadow = 3,
    GpuWavefrontQueueBsdf = 4,
    GpuWavefrontQueueNextRay = 5,
    GpuWavefrontQueueCount = 6,
};

struct GpuWavefrontQueueCounters {
    int num_queued[GpuWavefrontQueueCount];
};

static constexpr uint32_t kWavefrontBounceMask = 0xffu;
static constexpr uint32_t kWavefrontTransmissionShift = 8u;
static constexpr uint32_t kWavefrontTransparentShift = 16u;
static constexpr uint32_t kWavefrontPreviousDeltaBit = 1u << 24u;

__device__ bool wavefront_svgf_aov_enabled(GpuWavefrontSvgfAov aov) {
    return aov.albedo != nullptr && aov.emission != nullptr && aov.normal != nullptr &&
        aov.world_position != nullptr && aov.depth != nullptr && aov.object_id != nullptr;
}

__device__ Vec3 svgf_max_gpu(Vec3 v, float lo = 0.0f) {
    return {fmaxf(v.x, lo), fmaxf(v.y, lo), fmaxf(v.z, lo)};
}

__device__ uint32_t primary_object_id_gpu(const GpuHit& hit) {
    if (hit.mesh >= 0) return static_cast<uint32_t>(hit.mesh + 1);
    if (hit.sphere >= 0) return 0x20000000u | static_cast<uint32_t>(hit.sphere + 1);
    if (hit.triangle >= 0) return 0x40000000u | static_cast<uint32_t>(hit.triangle + 1);
    if (hit.material >= 0) return 0x80000000u | static_cast<uint32_t>(hit.material + 1);
    return 0u;
}

__device__ Vec3 material_base_color_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return mul(material.albedo, sample_texture_gpu(scene, material.texture_index, material_base_uv_gpu(material, uv)));
}

__device__ float svgf_albedo_triangle_uv_density_gpu(const GpuScene& scene, const GpuHit& hit) {
    if (hit.triangle < 0 || hit.triangle >= scene.triangle_count) {
        return 0.0f;
    }
    const GpuTriangle tri = scene.triangles[hit.triangle];
    const Vec3 e1 = sub(tri.v1, tri.v0);
    const Vec3 e2 = sub(tri.v2, tri.v0);
    const Vec3 world_cross = dcross(e1, e2);
    const float world_area2 = sqrtf(ddot(world_cross, world_cross));
    const float du1 = tri.uv1.x - tri.uv0.x;
    const float dv1 = tri.uv1.y - tri.uv0.y;
    const float du2 = tri.uv2.x - tri.uv0.x;
    const float dv2 = tri.uv2.y - tri.uv0.y;
    const float uv_area2 = fabsf(du1 * dv2 - dv1 * du2);
    if (world_area2 <= 1.0e-8f || uv_area2 <= 1.0e-10f) {
        return 0.0f;
    }
    return sqrtf(uv_area2 / world_area2);
}

__device__ float svgf_albedo_sphere_uv_density_gpu(const GpuScene& scene, const GpuHit& hit) {
    if (hit.sphere < 0 || hit.sphere >= scene.sphere_count) {
        return 0.0f;
    }
    const float radius = fmaxf(scene.spheres[hit.sphere].radius, 1.0e-5f);
    return 1.0f / (kPi * radius);
}

__device__ float svgf_albedo_lod_gpu(
    const GpuScene& scene,
    RenderSettings settings,
    const GpuMaterial& material,
    const GpuHit& hit,
    const Ray& ray) {
    if (material.texture_index < 0 || material.texture_index >= scene.texture_count) {
        return 0.0f;
    }
    const GpuTexture texture = scene.textures[material.texture_index];
    if (texture.width <= 0 || texture.height <= 0 || texture.mip_levels <= 1 || texture.object == 0) {
        return 0.0f;
    }

    float uv_per_world = svgf_albedo_triangle_uv_density_gpu(scene, hit);
    if (uv_per_world <= 0.0f) {
        uv_per_world = svgf_albedo_sphere_uv_density_gpu(scene, hit);
    }
    if (uv_per_world <= 0.0f) {
        return 0.0f;
    }

    const Vec3 forward = dnormalize(sub(scene.camera.target, scene.camera.position));
    const float depth = fmaxf(1.0e-4f, ddot(sub(hit.position, scene.camera.position), forward));
    const float image_height = fmaxf(1.0f, static_cast<float>(settings.height));
    const float pixel_world = 2.0f * tanf(scene.camera.fov_degrees * kPi / 360.0f) * depth / image_height;
    const float ndotv = fabsf(ddot(hit.normal, mul(ray.direction, -1.0f)));
    const float grazing_scale = 1.0f / fmaxf(0.25f, ndotv);
    const float uv_scale = fmaxf(fabsf(material.texture_scale.x), fabsf(material.texture_scale.y));
    const float texture_extent = static_cast<float>(texture.width > texture.height ? texture.width : texture.height);
    const float footprint_texels = pixel_world * uv_per_world * fmaxf(uv_scale, 1.0e-4f) *
        grazing_scale * texture_extent;
    const float lod = log2f(fmaxf(1.0f, footprint_texels)) + 0.75f;
    return dclamp(lod, 0.0f, static_cast<float>(texture.mip_levels - 1));
}

__device__ Vec3 material_base_color_svgf_gpu(
    const GpuScene& scene,
    RenderSettings settings,
    const GpuMaterial& material,
    const GpuHit& hit,
    const Ray& ray) {
    const Vec2 uv = material_base_uv_gpu(material, hit.uv);
    return mul(material.albedo, sample_texture_lod_gpu(scene, material.texture_index, uv,
        svgf_albedo_lod_gpu(scene, settings, material, hit, ray)));
}

__device__ void write_wavefront_svgf_default_aov(GpuWavefrontSvgfAov aov, int pixel) {
    if (!wavefront_svgf_aov_enabled(aov)) return;
    aov.albedo[pixel] = {1.0f, 1.0f, 1.0f};
    aov.emission[pixel] = {};
    aov.normal[pixel] = {};
    aov.world_position[pixel] = {};
    aov.depth[pixel] = kInfinity;
    aov.object_id[pixel] = 0u;
}

__device__ void write_wavefront_svgf_surface_aov(
    const GpuScene& scene,
    RenderSettings settings,
    GpuWavefrontSvgfAov aov,
    int pixel,
    const Ray& ray,
    const GpuHit& hit,
    const GpuMaterial& material,
    Vec3 material_emission) {
    if (!wavefront_svgf_aov_enabled(aov)) return;
    Vec3 hit_emission = add(hit.emission, material_emission);
    if (hit.triangle >= 0 && hit.triangle < scene.triangle_count) {
        const GpuTriangle light = scene.triangles[hit.triangle];
        hit_emission = emitted_radiance_gpu(light, material, light.emission,
            material_emission, light.light_double_sided != 0, ray.direction);
    }
    const Vec3 forward = dnormalize(sub(scene.camera.target, scene.camera.position));
    aov.albedo[pixel] = svgf_max_gpu(material_base_color_svgf_gpu(scene, settings, material, hit, ray), 0.001f);
    aov.emission[pixel] = svgf_max_gpu(hit_emission);
    aov.normal[pixel] = hit.normal;
    aov.world_position[pixel] = hit.position;
    aov.depth[pixel] = fmaxf(0.0f, ddot(sub(hit.position, scene.camera.position), forward));
    aov.object_id[pixel] = primary_object_id_gpu(hit);
}

__device__ GpuWavefrontPathRef wavefront_path_ref(GpuWavefrontPaths paths, int index) {
    return {
        paths.rays[index],
        paths.throughputs[index],
        paths.radiance[index],
        paths.previous_positions[index],
        paths.previous_bsdf_pdfs[index],
        paths.rngs[index],
        paths.pixels[index],
        paths.states[index],
    };
}

__device__ GpuWavefrontIntersectPathRef wavefront_intersect_path_ref(GpuWavefrontIntersectPaths paths, int index) {
    return {
        paths.rays[index],
        paths.throughputs[index],
        paths.radiance[index],
        paths.rngs[index],
        paths.pixels[index],
        paths.states[index],
    };
}

template <typename PathT>
__device__ int wavefront_bounce(const PathT& path) {
    return static_cast<int>(path.state & kWavefrontBounceMask);
}

template <typename PathT>
__device__ int wavefront_transmission_bounces(const PathT& path) {
    return static_cast<int>((path.state >> kWavefrontTransmissionShift) & 0xffu);
}

template <typename PathT>
__device__ int wavefront_transparent_steps(const PathT& path) {
    return static_cast<int>((path.state >> kWavefrontTransparentShift) & 0xffu);
}

template <typename PathT>
__device__ bool wavefront_previous_delta(const PathT& path) {
    return (path.state & kWavefrontPreviousDeltaBit) != 0u;
}

template <typename PathT>
__device__ void wavefront_set_previous_delta(PathT& path, bool value) {
    path.state = value ? (path.state | kWavefrontPreviousDeltaBit) : (path.state & ~kWavefrontPreviousDeltaBit);
}

template <typename PathT>
__device__ void wavefront_set_transparent_steps(PathT& path, int steps) {
    path.state = (path.state & ~(0xffu << kWavefrontTransparentShift)) |
        ((static_cast<uint32_t>(steps) & 0xffu) << kWavefrontTransparentShift);
}

template <typename PathT>
__device__ void wavefront_increment_bounce(PathT& path) {
    const uint32_t bounce = (path.state & kWavefrontBounceMask) + 1u;
    path.state = (path.state & ~kWavefrontBounceMask) | (bounce & kWavefrontBounceMask);
}

template <typename PathT>
__device__ void wavefront_increment_transmission_bounce(PathT& path) {
    const uint32_t transmission = ((path.state >> kWavefrontTransmissionShift) & 0xffu) + 1u;
    path.state = (path.state & ~(0xffu << kWavefrontTransmissionShift)) |
        ((transmission & 0xffu) << kWavefrontTransmissionShift);
}

__device__ void wavefront_append_queue(int* indices, int* count, int path_index) {
    const unsigned mask = __activemask();
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int leader = __ffs(mask) - 1;
    const int rank = __popc(mask & ((1u << lane) - 1u));
    const int warp_count = __popc(mask);
    int base = 0;
    if (lane == leader) {
        base = atomicAdd(count, warp_count);
    }
    base = __shfl_sync(mask, base, leader);
    indices[base + rank] = path_index;
}

template <typename PathT>
__device__ void finish_wavefront_path(PathT& path, Vec3* samples) {
    samples[path.pixel] = path.radiance;
}

__global__ void wavefront_initialize_kernel(
    const GpuScene* scene_ptr,
    RenderSettings settings,
    GpuWavefrontPaths paths,
    int* active_indices,
    GpuWavefrontQueueCounters* queue_counters,
    Vec3* samples,
    GpuWavefrontSvgfAov svgf_aov,
    int pixel_count,
    int sample_index,
    bool use_camera_jitter) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    if (idx == 0) {
        for (int i = 0; i < GpuWavefrontQueueCount; ++i) {
            queue_counters->num_queued[i] = 0;
        }
        queue_counters->num_queued[GpuWavefrontQueueActive] = pixel_count;
    }
    const int x = idx % settings.width;
    const int y = idx / settings.width;
    GpuWavefrontPathRef path = wavefront_path_ref(paths, idx);
    path.rng = make_pixel_seed(static_cast<uint32_t>(x), static_cast<uint32_t>(y),
        settings.frame_index ^ (0x9e3779b9u * static_cast<uint32_t>(sample_index + 1)));
    path.ray = use_camera_jitter
        ? camera_ray_with_sample(
            scene_ptr->camera,
            x,
            y,
            settings.width,
            settings.height,
            0.5f + settings.camera_jitter_x,
            0.5f + settings.camera_jitter_y)
        : camera_ray(scene_ptr->camera, x, y, settings.width, settings.height, path.rng);
    path.throughput = {1.0f, 1.0f, 1.0f};
    path.radiance = {};
    path.previous_position = {};
    path.previous_bsdf_pdf = 0.0f;
    path.pixel = idx;
    path.state = 0u;
    active_indices[idx] = idx;
    samples[idx] = {};
    write_wavefront_svgf_default_aov(svgf_aov, idx);
}

__global__ void wavefront_prepare_step_kernel(GpuWavefrontQueueCounters* queue_counters) {
    queue_counters->num_queued[GpuWavefrontQueueDirectLight] = 0;
    queue_counters->num_queued[GpuWavefrontQueueGi] = 0;
    queue_counters->num_queued[GpuWavefrontQueueShadow] = 0;
    queue_counters->num_queued[GpuWavefrontQueueBsdf] = 0;
    queue_counters->num_queued[GpuWavefrontQueueNextRay] = 0;
}

__global__ void wavefront_promote_next_kernel(GpuWavefrontQueueCounters* queue_counters) {
    queue_counters->num_queued[GpuWavefrontQueueActive] =
        queue_counters->num_queued[GpuWavefrontQueueNextRay];
}

template <bool AlphaVisibility, bool TwoLevel, bool Primary>
__global__ void wavefront_intersect_kernel(
    const GpuScene* scene_ptr,
    RenderSettings settings,
    GpuWavefrontIntersectPaths paths,
    GpuCompactHit* compact_hits,
    const int* active_indices,
    int* shade_indices,
    GpuWavefrontQueueCounters* queue_counters,
    Vec3* samples) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int active_count = queue_counters->num_queued[GpuWavefrontQueueActive];
    if (queue_index >= active_count) return;

    const GpuScene& scene = *scene_ptr;
    const int path_index = active_indices[queue_index];
    GpuWavefrontIntersectPathRef path = wavefront_intersect_path_ref(paths, path_index);
    const int shading_bounce = Primary ? 0 : wavefront_bounce(path) - wavefront_transmission_bounces(path);
    if (shading_bounce >= settings.max_bounces) {
        finish_wavefront_path(path, samples);
        return;
    }

    const float sample_clamp = shading_bounce == 0 ? 64.0f : 8.0f;
    Ray ray = path.ray;
    int transparent_steps = wavefront_transparent_steps(path);
    for (;;) {
        GpuCompactHit hit;
        if (!intersect_compact_gpu<TwoLevel>(scene, ray, hit)) {
            path.radiance = add(path.radiance, clamp_sample_radiance_gpu(
                mul(path.throughput, environment_radiance_gpu(scene, ray.direction, settings)), sample_clamp));
            finish_wavefront_path(path, samples);
            return;
        }
        const bool triangle_has_alpha = hit.triangle >= 0 && traversal_material_has_alpha(hit.material);
        const int material_index = hit.triangle >= 0 ? traversal_material_index(hit.material) : hit.material;
        if (material_index < 0 || material_index >= scene.material_count) {
            finish_wavefront_path(path, samples);
            return;
        }

        if constexpr (AlphaVisibility) {
            const bool test_visibility = hit.triangle < 0 || triangle_has_alpha;
            if (test_visibility) {
                const GpuMaterial material = scene.materials[material_index];
                const Vec2 uv = compact_hit_uv_gpu(scene, ray, hit);
                if (!material_visible_gpu(scene, material, uv, path.rng)) {
                    if (transparent_steps < 8) {
                        const Vec3 position = compact_hit_position_gpu(ray, hit);
                        ray = {add(position, mul(ray.direction, 0.002f)), ray.direction};
                        ++transparent_steps;
                        continue;
                    }
                    finish_wavefront_path(path, samples);
                    return;
                }
            }
        }

        path.ray = ray;
        wavefront_set_transparent_steps(path, 0);
        hit.material = material_index;
        compact_hits[path_index] = hit;
        wavefront_append_queue(shade_indices,
            &queue_counters->num_queued[GpuWavefrontQueueDirectLight], path_index);
        return;
    }
}

__global__ void wavefront_direct_light_kernel(
    const GpuScene* scene_ptr,
    RenderSettings settings,
    GpuWavefrontPaths paths,
    const GpuCompactHit* compact_hits,
    GpuHit* hits,
    const int* direct_indices,
    int* gi_indices,
    int* shadow_indices,
    int* bsdf_indices,
    GpuWavefrontQueueCounters* queue_counters,
    GpuWavefrontSvgfAov svgf_aov,
    Vec3* samples) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int direct_count = queue_counters->num_queued[GpuWavefrontQueueDirectLight];
    if (queue_index >= direct_count) return;

    const GpuScene& scene = *scene_ptr;
    const int path_index = direct_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    GpuHit hit;
    if (!fill_compact_hit_gpu(scene, path.ray, compact_hits[path_index], hit)) {
        finish_wavefront_path(path, samples);
        return;
    }
    const int shading_bounce = wavefront_bounce(path) - wavefront_transmission_bounces(path);
    const float sample_clamp = shading_bounce == 0 ? 64.0f : 8.0f;
    const GpuMaterial material = scene.materials[hit.material];

    hit.normal = apply_normal_map_gpu(scene, material, hit, path.ray.direction);
    const Vec3 material_emission = material_emission_gpu(scene, material, hit.uv);
    hit.emission = add(hit.emission, material_emission);
    if (shading_bounce == 0) {
        write_wavefront_svgf_surface_aov(scene, settings, svgf_aov, path.pixel, path.ray, hit, material, material_emission);
    }
    if (has_light_emission_gpu(hit.emission)) {
        const GpuTriangle light = hit.triangle >= 0 && hit.triangle < scene.triangle_count ? scene.triangles[hit.triangle] : GpuTriangle{};
        const GpuMaterial light_material = scene.materials[hit.material];
        const Vec3 emission = emitted_radiance_gpu(light, light_material, light.emission,
            material_emission, light.light_double_sided != 0, path.ray.direction);
        if (shading_bounce == 0 || wavefront_previous_delta(path)) {
            path.radiance = add(path.radiance, clamp_sample_radiance_gpu(mul(path.throughput, emission), sample_clamp));
        } else if (settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling &&
                   hit.triangle >= 0 && hit.triangle < scene.triangle_count) {
            const float light_pmf = scene.light_count > 0 ? 1.0f / static_cast<float>(scene.light_count) : 0.0f;
            const float light_pdf = light_pdf_solid_angle_gpu(light, light_material, path.previous_position, hit.position, light_pmf);
            path.radiance = add(path.radiance, clamp_sample_radiance_gpu(mul(mul(path.throughput, emission),
                mis_weight_gpu(path.previous_bsdf_pdf, light_pdf, static_cast<int>(settings.mis_heuristic))), sample_clamp));
        } else if (settings.sampling_mode == PathSamplingMode::Unidirectional) {
            path.radiance = add(path.radiance, clamp_sample_radiance_gpu(mul(path.throughput, emission), sample_clamp));
        }
        finish_wavefront_path(path, samples);
        return;
    }

    if (settings.use_lightmap && hit.has_lightmap && material_uses_lightmap_gi_gpu(settings, material)) {
        hits[path_index] = hit;
        wavefront_append_queue(gi_indices,
            &queue_counters->num_queued[GpuWavefrontQueueGi], path_index);
        return;
    }
    if (settings.use_irradiance_volume && material_uses_irradiance_volume_gi_gpu(settings, material)) {
        hits[path_index] = hit;
        wavefront_append_queue(gi_indices,
            &queue_counters->num_queued[GpuWavefrontQueueGi], path_index);
        return;
    }
    hits[path_index] = hit;
    if (settings.sampling_mode == PathSamplingMode::Unidirectional) {
        wavefront_append_queue(bsdf_indices,
            &queue_counters->num_queued[GpuWavefrontQueueBsdf], path_index);
        return;
    }
    wavefront_append_queue(shadow_indices,
        &queue_counters->num_queued[GpuWavefrontQueueShadow], path_index);
    wavefront_append_queue(bsdf_indices,
        &queue_counters->num_queued[GpuWavefrontQueueBsdf], path_index);
}

__global__ void wavefront_finalize_active_kernel(
    GpuWavefrontPaths paths,
    const int* active_indices,
    const GpuWavefrontQueueCounters* queue_counters,
    Vec3* samples) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int active_count = queue_counters->num_queued[GpuWavefrontQueueActive];
    if (queue_index >= active_count) return;
    const int path_index = active_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    finish_wavefront_path(path, samples);
}

__global__ void wavefront_gi_kernel(
    const GpuScene* scene_ptr,
    RenderSettings settings,
    GpuWavefrontPaths paths,
    const GpuHit* hits,
    const int* gi_indices,
    int* bsdf_indices,
    GpuWavefrontQueueCounters* queue_counters,
    Vec3* samples) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int gi_count = queue_counters->num_queued[GpuWavefrontQueueGi];
    if (queue_index >= gi_count) return;

    const GpuScene& scene = *scene_ptr;
    const int path_index = gi_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    const GpuHit hit = hits[path_index];
    const GpuMaterial material = scene.materials[hit.material];
    const int shading_bounce = wavefront_bounce(path) - wavefront_transmission_bounces(path);
    const float sample_clamp = shading_bounce == 0 ? 64.0f : 8.0f;
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    if (settings.use_lightmap && hit.has_lightmap && material_uses_lightmap_gi_gpu(settings, material)) {
        path.radiance = add(path.radiance, clamp_sample_radiance_gpu(
            mul(path.throughput, shade_material_from_lightmap_gpu(scene, settings, material, hit, wo, path.rng)), sample_clamp));
        finish_wavefront_path(path, samples);
        return;
    }
    if (settings.use_irradiance_volume && material_uses_irradiance_volume_gi_gpu(settings, material)) {
        path.radiance = add(path.radiance, clamp_sample_radiance_gpu(
            mul(path.throughput, shade_material_from_irradiance_volume_gpu(scene, settings, material, hit, wo, path.rng)), sample_clamp));
        finish_wavefront_path(path, samples);
        return;
    }
    wavefront_append_queue(bsdf_indices,
        &queue_counters->num_queued[GpuWavefrontQueueBsdf], path_index);
}

template <typename PathT>
__device__ void wavefront_sample_bsdf_to_next(
    const GpuScene& scene,
    RenderSettings settings,
    PathT& path,
    const GpuHit& hit,
    int path_index,
    int* next_indices,
    GpuWavefrontQueueCounters* queue_counters,
    Vec3* samples) {
    const int shading_bounce = wavefront_bounce(path) - wavefront_transmission_bounces(path);
    const GpuMaterial material = scene.materials[hit.material];
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    if (shading_bounce >= 3) {
        const float p = dclamp(fmaxf(path.throughput.x, fmaxf(path.throughput.y, path.throughput.z)), 0.05f, 0.95f);
        if (rng_float(path.rng) > p) {
            finish_wavefront_path(path, samples);
            return;
        }
        path.throughput = divv(path.throughput, p);
    }
    const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal, wo, hit.uv, hit.front_face, path.rng);
    if (!isfinite(sample.pdf) || sample.pdf <= 0.0f || !finite_vec_gpu(sample.weight) || ddot(sample.weight, sample.weight) <= 0.0f) {
        finish_wavefront_path(path, samples);
        return;
    }
    path.previous_position = hit.position;
    path.previous_bsdf_pdf = sample.pdf;
    wavefront_set_previous_delta(path, sample.delta);
    if (sample.delta && material_transmission_gpu(scene, material, hit.uv) > 0.5f && wavefront_transmission_bounces(path) < 12) {
        wavefront_increment_transmission_bounce(path);
    }
    const float offset_side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
    path.ray = {add(hit.position, mul(hit.normal, 0.001f * offset_side)), sample.direction};
    path.throughput = mul(path.throughput, sample.weight);
    wavefront_increment_bounce(path);
    wavefront_append_queue(next_indices,
        &queue_counters->num_queued[GpuWavefrontQueueNextRay], path_index);
}

__global__ void wavefront_direct_visibility_kernel(
    const GpuScene* scene_ptr,
    RenderSettings settings,
    GpuWavefrontPaths paths,
    const GpuHit* hits,
    const int* shadow_indices,
    GpuWavefrontQueueCounters* queue_counters,
    Vec3* /*samples*/) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;

    const GpuScene& scene = *scene_ptr;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    const GpuHit hit = hits[path_index];
    const int shading_bounce = wavefront_bounce(path) - wavefront_transmission_bounces(path);
    const float sample_clamp = shading_bounce == 0 ? 64.0f : 8.0f;
    const GpuMaterial material = scene.materials[hit.material];
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    path.radiance = add(path.radiance, clamp_sample_radiance_gpu(
        mul(path.throughput, estimate_direct_gpu(scene, hit, material, wo, path.rng, settings)), sample_clamp));
}

__global__ void wavefront_bsdf_sample_kernel(
    const GpuScene* scene_ptr,
    RenderSettings settings,
    GpuWavefrontPaths paths,
    const GpuHit* hits,
    const int* bsdf_indices,
    int* next_indices,
    GpuWavefrontQueueCounters* queue_counters,
    Vec3* samples) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int bsdf_count = queue_counters->num_queued[GpuWavefrontQueueBsdf];
    if (queue_index >= bsdf_count) return;

    const GpuScene& scene = *scene_ptr;
    const int path_index = bsdf_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    const GpuHit hit = hits[path_index];
    wavefront_sample_bsdf_to_next(scene, settings, path, hit, path_index, next_indices, queue_counters, samples);
}

__global__ void wavefront_resolve_kernel(
    RenderSettings settings,
    const Vec3* sample_sum,
    Vec3* accumulation,
    uint32_t* rgba,
    int pixel_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    const Vec3 sample = divv(sample_sum[idx], static_cast<float>(settings.samples_per_pixel));
    accumulation[idx] = add(accumulation[idx], sample);
    rgba[idx] = rgba8_gpu(divv(accumulation[idx], static_cast<float>(settings.frame_index + 1u)));
}

__global__ void wavefront_svgf_radiance_kernel(
    RenderSettings settings,
    const Vec3* sample_sum,
    Vec3* radiance,
    int pixel_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    radiance[idx] = divv(sample_sum[idx], static_cast<float>(settings.samples_per_pixel));
}

__global__ void wavefront_accumulate_sample_kernel(
    const Vec3* samples,
    Vec3* sample_sum,
    int pixel_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) return;
    sample_sum[idx] = add(sample_sum[idx], samples[idx]);
}

__device__ float svgf_luminance_gpu(Vec3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ Ray center_camera_ray_gpu(const Camera& camera, int x, int y, int width, int height) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float half_height = tanf(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = dnormalize(sub(camera.target, camera.position));
    const float right_sign = camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const Vec3 right = mul(dnormalize(dcross(forward, camera.up)), right_sign);
    const Vec3 up = mul(dcross(right, forward), right_sign);
    const float u = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f) * half_width;
    const float v = (1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.0f) * half_height;
    return {camera.position, dnormalize(add(add(forward, mul(right, u)), mul(up, v)))};
}

__device__ Ray jittered_camera_ray_gpu(const Camera& camera, int x, int y, int width, int height, float jitter_x, float jitter_y) {
    return camera_ray_with_sample(camera, x, y, width, height, 0.5f + jitter_x, 0.5f + jitter_y);
}

__device__ void write_primary_aov_gpu(
    const GpuScene& scene,
    RenderSettings settings,
    int x,
    int y,
    int idx,
    Vec3* albedo,
    Vec3* emission,
    Vec3* normal,
    Vec3* world_position,
    float* depth,
    uint32_t* object_id) {
    albedo[idx] = {1.0f, 1.0f, 1.0f};
    emission[idx] = {};
    normal[idx] = {};
    world_position[idx] = {};
    depth[idx] = kInfinity;
    object_id[idx] = 0u;

    Ray ray = jittered_camera_ray_gpu(scene.camera, x, y, settings.width, settings.height, settings.camera_jitter_x, settings.camera_jitter_y);
    uint32_t visibility_rng = make_pixel_seed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), settings.frame_index ^ 0x9e3779b9u);
    for (int transparent_step = 0; transparent_step < 8; ++transparent_step) {
        GpuHit hit;
        if (!intersect_gpu(scene, ray, hit)) return;
        if (hit.material < 0 || hit.material >= scene.material_count) return;
        const GpuMaterial material = scene.materials[hit.material];
        if (!material_visible_gpu(scene, material, hit.uv, visibility_rng)) {
            ray = {add(hit.position, mul(ray.direction, 0.002f)), ray.direction};
            continue;
        }

        hit.normal = apply_normal_map_gpu(scene, material, hit, ray.direction);
        Vec3 hit_emission = material_emission_gpu(scene, material, hit.uv);
        if (hit.triangle >= 0 && hit.triangle < scene.triangle_count) {
            const GpuTriangle light = scene.triangles[hit.triangle];
            hit_emission = emitted_radiance_gpu(light, material, light.emission, hit_emission, light.light_double_sided != 0, ray.direction);
        }
        const Vec3 forward = dnormalize(sub(scene.camera.target, scene.camera.position));
        albedo[idx] = svgf_max_gpu(material_base_color_svgf_gpu(scene, settings, material, hit, ray), 0.001f);
        emission[idx] = svgf_max_gpu(hit_emission);
        normal[idx] = hit.normal;
        world_position[idx] = hit.position;
        depth[idx] = fmaxf(0.0f, ddot(sub(hit.position, scene.camera.position), forward));
        object_id[idx] = primary_object_id_gpu(hit);
        return;
    }
}

__global__ void render_svgf_input_kernel(
    const GpuScene* scene_ptr,
    RenderSettings settings,
    Vec3* radiance,
    Vec3* albedo,
    Vec3* emission,
    Vec3* normal,
    Vec3* world_position,
    float* depth,
    uint32_t* object_id,
    bool write_primary_aov) {
    const GpuScene& scene = *scene_ptr;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;
    const int idx = y * settings.width + x;
    uint32_t rng = make_pixel_seed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), settings.frame_index);
    Vec3 sample{};
    for (int s = 0; s < settings.samples_per_pixel; ++s) {
        sample = add(sample, trace_gpu_with_irradiance_probe_debug(
            scene,
            jittered_camera_ray_gpu(scene.camera, x, y, settings.width, settings.height, settings.camera_jitter_x, settings.camera_jitter_y),
            rng,
            settings));
    }
    radiance[idx] = divv(sample, static_cast<float>(settings.samples_per_pixel));
    if (write_primary_aov) {
        write_primary_aov_gpu(scene, settings, x, y, idx, albedo, emission, normal, world_position, depth, object_id);
    }
}

__device__ bool svgf_has_surface_gpu(const float* depth, const uint32_t* object_id, const Vec3* normal, int idx) {
    return object_id[idx] != 0u && isfinite(depth[idx]) && depth[idx] > 0.0f && ddot(normal[idx], normal[idx]) > 0.0f;
}

__device__ Vec3 svgf_demodulate_gpu(Vec3 radiance, Vec3 emission, Vec3 albedo) {
    const Vec3 lighting = svgf_max_gpu(sub(radiance, emission));
    return {
        lighting.x / fmaxf(albedo.x, 0.001f),
        lighting.y / fmaxf(albedo.y, 0.001f),
        lighting.z / fmaxf(albedo.z, 0.001f),
    };
}

__device__ bool project_to_pixel_gpu(const Camera& camera, Vec3 p, int width, int height, float jitter_x, float jitter_y, float& px, float& py, float& linear_depth) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float half_height = tanf(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = dnormalize(sub(camera.target, camera.position));
    const float right_sign = camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const Vec3 right = mul(dnormalize(dcross(forward, camera.up)), right_sign);
    const Vec3 up = mul(dcross(right, forward), right_sign);
    const Vec3 rel = sub(p, camera.position);
    linear_depth = ddot(rel, forward);
    if (!isfinite(linear_depth) || linear_depth <= 0.0001f) return false;
    const float ndc_x = ddot(rel, right) / (linear_depth * half_width);
    const float ndc_y = ddot(rel, up) / (linear_depth * half_height);
    if (!isfinite(ndc_x) || !isfinite(ndc_y) || ndc_x < -1.25f || ndc_x > 1.25f || ndc_y < -1.25f || ndc_y > 1.25f) return false;
    px = (ndc_x * 0.5f + 0.5f) * static_cast<float>(width) - jitter_x;
    py = (0.5f - ndc_y * 0.5f) * static_cast<float>(height) - jitter_y;
    return px >= 0.0f && py >= 0.0f && px < static_cast<float>(width) && py < static_cast<float>(height);
}

__global__ void svgf_temporal_kernel(
    RenderSettings settings,
    Camera previous_camera,
    float previous_jitter_x,
    float previous_jitter_y,
    int history_valid,
    Vec3* radiance,
    Vec3* albedo,
    Vec3* emission,
    Vec3* normal,
    Vec3* world_position,
    float* depth,
    uint32_t* object_id,
    Vec3* history_illumination,
    float* history_moment1,
    float* history_moment2,
    float* history_length,
    Vec3* history_normal,
    Vec3* history_world_position,
    float* history_depth,
    uint32_t* history_object_id,
    Vec3* temporal_illumination,
    float* temporal_moment1,
    float* temporal_moment2,
    float* temporal_variance,
    float* temporal_history_length) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;
    const int idx = y * settings.width + x;
    if (!svgf_has_surface_gpu(depth, object_id, normal, idx)) {
        temporal_illumination[idx] = radiance[idx];
        temporal_moment1[idx] = svgf_luminance_gpu(radiance[idx]);
        temporal_moment2[idx] = temporal_moment1[idx] * temporal_moment1[idx];
        temporal_variance[idx] = 0.0f;
        temporal_history_length[idx] = 0.0f;
        return;
    }

    const Vec3 illum = svgf_demodulate_gpu(radiance[idx], emission[idx], albedo[idx]);
    const float lum = svgf_luminance_gpu(illum);
    const float m1 = lum;
    const float m2 = lum * lum;
    bool reprojected = false;
    Vec3 prev_illum{};
    float prev_m1 = 0.0f;
    float prev_m2 = 0.0f;
    float prev_history = 0.0f;

    if (history_valid) {
        float px = 0.0f;
        float py = 0.0f;
        float prev_linear_depth = 0.0f;
        if (project_to_pixel_gpu(previous_camera, world_position[idx], settings.width, settings.height, previous_jitter_x, previous_jitter_y, px, py, prev_linear_depth)) {
            const int prev_x = iclamp_gpu(static_cast<int>(floorf(px)), 0, settings.width - 1);
            const int prev_y = iclamp_gpu(static_cast<int>(floorf(py)), 0, settings.height - 1);
            const int prev_idx = prev_y * settings.width + prev_x;
            const float depth_tolerance = fmaxf(0.02f, 0.06f * fmaxf(depth[idx], 1.0f));
            if (history_length[prev_idx] > 0.0f &&
                history_object_id[prev_idx] == object_id[idx] &&
                fabsf(history_depth[prev_idx] - prev_linear_depth) <= depth_tolerance &&
                ddot(history_normal[prev_idx], normal[idx]) >= 0.72f) {
                reprojected = true;
                prev_illum = history_illumination[prev_idx];
                prev_m1 = history_moment1[prev_idx];
                prev_m2 = history_moment2[prev_idx];
                prev_history = history_length[prev_idx];
            }
        }
    }

    const float h = fminf(32.0f, reprojected ? prev_history + 1.0f : 1.0f);
    const float alpha = reprojected ? fmaxf(settings.svgf_alpha, 1.0f / h) : 1.0f;
    const float moments_alpha = reprojected ? fmaxf(settings.svgf_moments_alpha, 1.0f / h) : 1.0f;
    temporal_illumination[idx] = reprojected ? dlerp(prev_illum, illum, alpha) : illum;
    temporal_moment1[idx] = reprojected ? prev_m1 * (1.0f - moments_alpha) + m1 * moments_alpha : m1;
    temporal_moment2[idx] = reprojected ? prev_m2 * (1.0f - moments_alpha) + m2 * moments_alpha : m2;
    temporal_variance[idx] = fmaxf(0.0f, temporal_moment2[idx] - temporal_moment1[idx] * temporal_moment1[idx]);
    temporal_history_length[idx] = h;
}

__device__ float svgf_edge_weight_gpu(
    const Vec3* illumination,
    const float* variance,
    const Vec3* normal,
    const float* depth,
    const uint32_t* object_id,
    int width,
    int height,
    int cx,
    int cy,
    int sx,
    int sy,
    RenderSettings settings) {
    if (sx < 0 || sy < 0 || sx >= width || sy >= height) return 0.0f;
    const int c = cy * width + cx;
    const int s = sy * width + sx;
    if (object_id[c] != object_id[s]) return 0.0f;
    if (!svgf_has_surface_gpu(depth, object_id, normal, c) || !svgf_has_surface_gpu(depth, object_id, normal, s)) {
        return c == s ? 1.0f : 0.0f;
    }
    const float normal_weight = expf(-fmaxf(0.0f, 1.0f - ddot(normal[c], normal[s])) * settings.svgf_phi_normal);
    const float depth_scale = fmaxf(0.01f, settings.svgf_phi_depth * 0.015f * fmaxf(depth[c], 1.0f));
    const float depth_weight = expf(-fabsf(depth[c] - depth[s]) / depth_scale);
    const float color_scale = fmaxf(0.01f, settings.svgf_phi_color * sqrtf(fmaxf(variance[c], 0.0f)) + 0.001f);
    const float color_weight = expf(-fabsf(svgf_luminance_gpu(illumination[c]) - svgf_luminance_gpu(illumination[s])) / color_scale);
    return normal_weight * depth_weight * color_weight;
}

__global__ void svgf_atrous_kernel(
    RenderSettings settings,
    int step,
    Vec3* input_illumination,
    float* input_variance,
    Vec3* normal,
    float* depth,
    uint32_t* object_id,
    Vec3* output_illumination,
    float* output_variance) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;
    const int idx = y * settings.width + x;
    if (!svgf_has_surface_gpu(depth, object_id, normal, idx)) {
        output_illumination[idx] = input_illumination[idx];
        output_variance[idx] = input_variance[idx];
        return;
    }
    const float kernel[5] = {1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};
    Vec3 sum{};
    float var_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            const int sx = x + kx * step;
            const int sy = y + ky * step;
            const float weight = kernel[kx + 2] * kernel[ky + 2] *
                svgf_edge_weight_gpu(input_illumination, input_variance, normal, depth, object_id, settings.width, settings.height, x, y, sx, sy, settings);
            if (weight <= 0.0f) continue;
            const int s = sy * settings.width + sx;
            sum = add(sum, mul(input_illumination[s], weight));
            var_sum += input_variance[s] * weight;
            weight_sum += weight;
        }
    }
    if (weight_sum > 0.0f) {
        output_illumination[idx] = divv(sum, weight_sum);
        output_variance[idx] = var_sum / weight_sum;
    } else {
        output_illumination[idx] = input_illumination[idx];
        output_variance[idx] = input_variance[idx];
    }
}

__device__ Vec3 svgf_debug_color_gpu(
    RenderSettings settings,
    int idx,
    Vec3 filtered_illumination,
    float variance,
    float history_len,
    Vec3 radiance,
    Vec3 albedo,
    Vec3 emission,
    Vec3 normal,
    float depth) {
    if (settings.svgf_debug_view == DenoiserDebugView::Raw) return radiance;
    if (settings.svgf_debug_view == DenoiserDebugView::Albedo) return albedo;
    if (settings.svgf_debug_view == DenoiserDebugView::Normal) return add(mul(normal, 0.5f), Vec3{0.5f, 0.5f, 0.5f});
    if (settings.svgf_debug_view == DenoiserDebugView::Depth) {
        const float d = isfinite(depth) ? depth : 0.0f;
        const float v = d / (d + 10.0f);
        return {v, v, v};
    }
    if (settings.svgf_debug_view == DenoiserDebugView::Variance) {
        const float v = dclamp(sqrtf(fmaxf(0.0f, variance)) * 0.25f, 0.0f, 1.0f);
        return {v, v, v};
    }
    if (settings.svgf_debug_view == DenoiserDebugView::HistoryLength) {
        const float v = dclamp(history_len / 32.0f, 0.0f, 1.0f);
        return {v, v, v};
    }
    return add(mul(filtered_illumination, albedo), emission);
}

__global__ void svgf_resolve_kernel(
    RenderSettings settings,
    Vec3* radiance,
    Vec3* albedo,
    Vec3* emission,
    Vec3* normal,
    Vec3* world_position,
    float* depth,
    uint32_t* object_id,
    Vec3* filtered_illumination,
    float* filtered_variance,
    float* temporal_moment1,
    float* temporal_moment2,
    float* temporal_history_length,
    Vec3* history_illumination,
    float* history_moment1,
    float* history_moment2,
    float* history_length,
    Vec3* history_normal,
    Vec3* history_world_position,
    float* history_depth,
    uint32_t* history_object_id,
    Vec3* final_color) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;
    const int idx = y * settings.width + x;
    if (!svgf_has_surface_gpu(depth, object_id, normal, idx)) {
        final_color[idx] = radiance[idx];
        history_illumination[idx] = radiance[idx];
        history_moment1[idx] = svgf_luminance_gpu(radiance[idx]);
        history_moment2[idx] = history_moment1[idx] * history_moment1[idx];
        history_length[idx] = 0.0f;
        history_normal[idx] = {};
        history_world_position[idx] = {};
        history_depth[idx] = kInfinity;
        history_object_id[idx] = 0u;
        return;
    }
    const Vec3 color = svgf_debug_color_gpu(
        settings, idx, filtered_illumination[idx], filtered_variance[idx], temporal_history_length[idx],
        radiance[idx], albedo[idx], emission[idx], normal[idx], depth[idx]);
    final_color[idx] = color;
    history_illumination[idx] = filtered_illumination[idx];
    history_moment1[idx] = temporal_moment1[idx];
    history_moment2[idx] = temporal_moment2[idx];
    history_length[idx] = temporal_history_length[idx];
    history_normal[idx] = normal[idx];
    history_world_position[idx] = world_position[idx];
    history_depth[idx] = depth[idx];
    history_object_id[idx] = object_id[idx];
}

__device__ Vec3 taa_min_gpu(Vec3 a, Vec3 b) {
    return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

__device__ Vec3 taa_max_gpu(Vec3 a, Vec3 b) {
    return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

__device__ Vec3 taa_clamp_gpu(Vec3 v, Vec3 lo, Vec3 hi) {
    return {
        dclamp(v.x, lo.x, hi.x),
        dclamp(v.y, lo.y, hi.y),
        dclamp(v.z, lo.z, hi.z),
    };
}

__device__ bool temporal_aa_enabled_gpu(RenderSettings settings) {
    return settings.denoiser_mode == DenoiserMode::Svgf &&
        settings.svgf_debug_view == DenoiserDebugView::Final &&
        settings.antialiasing_mode == AntialiasingMode::TAA;
}

__device__ bool post_aa_enabled_gpu(RenderSettings settings) {
    return settings.denoiser_mode == DenoiserMode::Svgf &&
        settings.svgf_debug_view == DenoiserDebugView::Final &&
        settings.antialiasing_mode == AntialiasingMode::StablePostAA;
}

__device__ Vec3 sample_bilinear_color_gpu(const Vec3* color, int width, int height, float x, float y) {
    const int x0 = iclamp_gpu(static_cast<int>(floorf(x)), 0, width - 1);
    const int y0 = iclamp_gpu(static_cast<int>(floorf(y)), 0, height - 1);
    const int x1 = iclamp_gpu(x0 + 1, 0, width - 1);
    const int y1 = iclamp_gpu(y0 + 1, 0, height - 1);
    const float tx = dclamp(x - static_cast<float>(x0), 0.0f, 1.0f);
    const float ty = dclamp(y - static_cast<float>(y0), 0.0f, 1.0f);
    const Vec3 a = dlerp(color[y0 * width + x0], color[y0 * width + x1], tx);
    const Vec3 b = dlerp(color[y1 * width + x0], color[y1 * width + x1], tx);
    return dlerp(a, b, ty);
}

__device__ Vec3 rgb_to_ycocg_gpu(Vec3 c) {
    const float co = c.x - c.z;
    const float t = c.z + co * 0.5f;
    const float cg = c.y - t;
    const float y = t + cg * 0.5f;
    return {y, co, cg};
}

__device__ Vec3 ycocg_to_rgb_gpu(Vec3 c) {
    const float t = c.x - c.z * 0.5f;
    const float g = c.z + t;
    const float b = t - c.y * 0.5f;
    const float r = b + c.y;
    return {r, g, b};
}

__device__ void taa_neighborhood_ycocg_stats_gpu(const Vec3* color, int width, int height, int x, int y, Vec3& lo, Vec3& hi, Vec3& mean, Vec3& sigma) {
    lo = {kInfinity, kInfinity, kInfinity};
    hi = {-kInfinity, -kInfinity, -kInfinity};
    mean = {};
    Vec3 sum2{};
    float count = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        const int sy = iclamp_gpu(y + dy, 0, height - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            const int sx = iclamp_gpu(x + dx, 0, width - 1);
            const Vec3 c = rgb_to_ycocg_gpu(color[sy * width + sx]);
            lo = taa_min_gpu(lo, c);
            hi = taa_max_gpu(hi, c);
            mean = add(mean, c);
            sum2 = add(sum2, mul(c, c));
            count += 1.0f;
        }
    }
    mean = divv(mean, count);
    const Vec3 variance = svgf_max_gpu(sub(divv(sum2, count), mul(mean, mean)));
    sigma = {sqrtf(variance.x), sqrtf(variance.y), sqrtf(variance.z)};
}

__device__ Vec3 taa_clip_history_gpu(const Vec3* current_color, int width, int height, int x, int y, Vec3 history_color) {
    Vec3 lo;
    Vec3 hi;
    Vec3 mean;
    Vec3 sigma;
    taa_neighborhood_ycocg_stats_gpu(current_color, width, height, x, y, lo, hi, mean, sigma);
    lo = taa_max_gpu(lo, sub(mean, mul(sigma, 1.25f)));
    hi = taa_min_gpu(hi, add(mean, mul(sigma, 1.25f)));
    return ycocg_to_rgb_gpu(taa_clamp_gpu(rgb_to_ycocg_gpu(history_color), lo, hi));
}

__device__ bool taa_valid_history_sample_gpu(
    const Vec3* normal,
    const float* depth,
    const uint32_t* object_id,
    const Vec3* history_normal,
    const float* history_depth,
    const uint32_t* history_object_id,
    int idx,
    int prev_idx,
    bool surface,
    float previous_linear_depth,
    bool allow_object_mismatch) {
    const bool same_object = history_object_id[prev_idx] == object_id[idx];
    if (!same_object) return allow_object_mismatch;
    if (!surface) return true;
    const float depth_tolerance = fmaxf(0.02f, 0.06f * fmaxf(depth[idx], 1.0f));
    return fabsf(history_depth[prev_idx] - previous_linear_depth) <= depth_tolerance &&
        ddot(history_normal[prev_idx], normal[idx]) >= 0.72f;
}

__device__ bool taa_object_edge_gpu(const uint32_t* object_id, int width, int height, int x, int y) {
    const int center = y * width + x;
    const int left = y * width + iclamp_gpu(x - 1, 0, width - 1);
    const int right = y * width + iclamp_gpu(x + 1, 0, width - 1);
    const int up = iclamp_gpu(y - 1, 0, height - 1) * width + x;
    const int down = iclamp_gpu(y + 1, 0, height - 1) * width + x;
    return object_id[center] != object_id[left] ||
        object_id[center] != object_id[right] ||
        object_id[center] != object_id[up] ||
        object_id[center] != object_id[down];
}

__device__ bool taa_sample_history_bilinear_gpu(
    RenderSettings settings,
    int idx,
    bool surface,
    float previous_x,
    float previous_y,
    float previous_linear_depth,
    const Vec3* normal,
    const float* depth,
    const uint32_t* object_id,
    const Vec3* history_color,
    const Vec3* history_normal,
    const float* history_depth,
    const uint32_t* history_object_id,
    const float* history_length,
    bool allow_object_mismatch,
    Vec3& previous_color,
    float& previous_history_length) {
    const int x0 = iclamp_gpu(static_cast<int>(floorf(previous_x)), 0, settings.width - 1);
    const int y0 = iclamp_gpu(static_cast<int>(floorf(previous_y)), 0, settings.height - 1);
    const int x1 = iclamp_gpu(x0 + 1, 0, settings.width - 1);
    const int y1 = iclamp_gpu(y0 + 1, 0, settings.height - 1);
    const float tx = dclamp(previous_x - static_cast<float>(x0), 0.0f, 1.0f);
    const float ty = dclamp(previous_y - static_cast<float>(y0), 0.0f, 1.0f);
    const int xs[2] = {x0, x1};
    const int ys[2] = {y0, y1};
    const float wx[2] = {1.0f - tx, tx};
    const float wy[2] = {1.0f - ty, ty};
    Vec3 sum{};
    float length_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int yy = 0; yy < 2; ++yy) {
        for (int xx = 0; xx < 2; ++xx) {
            const float w = wx[xx] * wy[yy];
            const int prev_idx = ys[yy] * settings.width + xs[xx];
            if (w <= 0.0f ||
                !taa_valid_history_sample_gpu(
                    normal,
                    depth,
                    object_id,
                    history_normal,
                    history_depth,
                    history_object_id,
                    idx,
                    prev_idx,
                    surface,
                    previous_linear_depth,
                    allow_object_mismatch)) {
                continue;
            }
            sum = add(sum, mul(history_color[prev_idx], w));
            length_sum += history_length[prev_idx] * w;
            weight_sum += w;
        }
    }
    if (weight_sum <= 0.0f) return false;
    previous_color = divv(sum, weight_sum);
    previous_history_length = length_sum / weight_sum;
    return true;
}

__device__ Vec3 taa_post_aa_gpu(RenderSettings settings, const Vec3* current_color, const uint32_t* object_id, int x, int y, Vec3 color) {
    const int nw = iclamp_gpu(y - 1, 0, settings.height - 1) * settings.width + iclamp_gpu(x - 1, 0, settings.width - 1);
    const int ne = iclamp_gpu(y - 1, 0, settings.height - 1) * settings.width + iclamp_gpu(x + 1, 0, settings.width - 1);
    const int left = y * settings.width + iclamp_gpu(x - 1, 0, settings.width - 1);
    const int right = y * settings.width + iclamp_gpu(x + 1, 0, settings.width - 1);
    const int up = iclamp_gpu(y - 1, 0, settings.height - 1) * settings.width + x;
    const int down = iclamp_gpu(y + 1, 0, settings.height - 1) * settings.width + x;
    const int sw = iclamp_gpu(y + 1, 0, settings.height - 1) * settings.width + iclamp_gpu(x - 1, 0, settings.width - 1);
    const int se = iclamp_gpu(y + 1, 0, settings.height - 1) * settings.width + iclamp_gpu(x + 1, 0, settings.width - 1);
    const int center = y * settings.width + x;
    const float lc = svgf_luminance_gpu(current_color[center]);
    const float lnw = svgf_luminance_gpu(current_color[nw]);
    const float lne = svgf_luminance_gpu(current_color[ne]);
    const float ll = svgf_luminance_gpu(current_color[left]);
    const float lr = svgf_luminance_gpu(current_color[right]);
    const float lu = svgf_luminance_gpu(current_color[up]);
    const float ld = svgf_luminance_gpu(current_color[down]);
    const float lsw = svgf_luminance_gpu(current_color[sw]);
    const float lse = svgf_luminance_gpu(current_color[se]);
    const float lmin = fminf(fminf(fminf(fminf(fminf(fminf(fminf(fminf(lc, lnw), lne), ll), lr), lu), ld), lsw), lse);
    const float lmax = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(lc, lnw), lne), ll), lr), lu), ld), lsw), lse);
    const bool geometry_edge =
        object_id[center] != object_id[left] ||
        object_id[center] != object_id[right] ||
        object_id[center] != object_id[up] ||
        object_id[center] != object_id[down];
    const float range = lmax - lmin;
    if (!geometry_edge && range < 0.025f) return color;
    float dir_x = -((lnw + lne) - (lsw + lse));
    float dir_y = ((lnw + lsw) - (lne + lse));
    const float dir_reduce = fmaxf((lnw + lne + lsw + lse) * 0.03125f, 0.0078125f);
    const float inv_min = 1.0f / (fminf(fabsf(dir_x), fabsf(dir_y)) + dir_reduce);
    dir_x = dclamp(dir_x * inv_min, -8.0f, 8.0f);
    dir_y = dclamp(dir_y * inv_min, -8.0f, 8.0f);
    const float px = static_cast<float>(x);
    const float py = static_cast<float>(y);
    const Vec3 rgb_a = mul(add(
        sample_bilinear_color_gpu(current_color, settings.width, settings.height, px + dir_x * (-1.0f / 6.0f), py + dir_y * (-1.0f / 6.0f)),
        sample_bilinear_color_gpu(current_color, settings.width, settings.height, px + dir_x * (1.0f / 6.0f), py + dir_y * (1.0f / 6.0f))), 0.5f);
    const Vec3 rgb_b = add(mul(rgb_a, 0.5f), mul(add(
        sample_bilinear_color_gpu(current_color, settings.width, settings.height, px + dir_x * -0.5f, py + dir_y * -0.5f),
        sample_bilinear_color_gpu(current_color, settings.width, settings.height, px + dir_x * 0.5f, py + dir_y * 0.5f)), 0.25f));
    const float luma_b = svgf_luminance_gpu(rgb_b);
    const Vec3 fxaa_color = (luma_b < lmin || luma_b > lmax) ? rgb_a : rgb_b;
    const float strength = geometry_edge ? 0.92f : dclamp((range - 0.025f) / 0.20f, 0.0f, 0.75f);
    return dlerp(color, fxaa_color, strength);
}

__global__ void taa_resolve_kernel(
    RenderSettings settings,
    Camera previous_camera,
    float previous_jitter_x,
    float previous_jitter_y,
    int history_valid,
    const Vec3* current_color,
    const Vec3* normal,
    const Vec3* world_position,
    const float* depth,
    const uint32_t* object_id,
    Vec3* history_color,
    Vec3* history_normal,
    Vec3* history_world_position,
    float* history_depth,
    uint32_t* history_object_id,
    float* history_length,
    uint32_t* rgba) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= settings.width || y >= settings.height) return;
    const int idx = y * settings.width + x;

    Vec3 color = current_color[idx];
    float resolved_history_length = 1.0f;
    const bool use_taa = temporal_aa_enabled_gpu(settings);
    if (use_taa && history_valid) {
        bool valid = false;
        Vec3 previous{};
        const bool surface = svgf_has_surface_gpu(depth, object_id, normal, idx);
        float px = 0.0f;
        float py = 0.0f;
        float previous_linear_depth = 0.0f;
        const bool allow_object_mismatch =
            (settings.camera_jitter_x != previous_jitter_x || settings.camera_jitter_y != previous_jitter_y) &&
            settings.dirty == RenderDirty::None &&
            taa_object_edge_gpu(object_id, settings.width, settings.height, x, y);
        if (surface) {
            valid = project_to_pixel_gpu(previous_camera, world_position[idx], settings.width, settings.height, previous_jitter_x, previous_jitter_y, px, py, previous_linear_depth);
        } else if (object_id[idx] == 0u) {
            px = static_cast<float>(x) + settings.camera_jitter_x - previous_jitter_x;
            py = static_cast<float>(y) + settings.camera_jitter_y - previous_jitter_y;
            valid = px >= 0.0f && py >= 0.0f && px < static_cast<float>(settings.width) && py < static_cast<float>(settings.height);
        }
        if (valid) {
            float previous_history_length = 0.0f;
            if (taa_sample_history_bilinear_gpu(
                    settings,
                    idx,
                    surface,
                    px,
                    py,
                    previous_linear_depth,
                    normal,
                    depth,
                    object_id,
                    history_color,
                    history_normal,
                    history_depth,
                    history_object_id,
                    history_length,
                    allow_object_mismatch,
                    previous,
                    previous_history_length)) {
                previous = taa_clip_history_gpu(current_color, settings.width, settings.height, x, y, previous);
                resolved_history_length = fminf(32.0f, previous_history_length + 1.0f);
                const float min_alpha = settings.dirty == RenderDirty::None ? 0.04f : 0.25f;
                const float alpha = fmaxf(min_alpha, 1.0f / resolved_history_length);
                color = dlerp(previous, color, alpha);
            }
        }
    }
    if (post_aa_enabled_gpu(settings)) {
        color = taa_post_aa_gpu(settings, current_color, object_id, x, y, color);
    }

    rgba[idx] = rgba8_gpu(color);
    history_color[idx] = color;
    history_normal[idx] = normal[idx];
    history_world_position[idx] = world_position[idx];
    history_depth[idx] = depth[idx];
    history_object_id[idx] = object_id[idx];
    history_length[idx] = resolved_history_length;
}

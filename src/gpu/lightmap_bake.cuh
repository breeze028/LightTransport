// GPU lightmap bake kernels.
// Included from cuda_path_tracer.cu inside the anonymous namespace.
// The host orchestrator is defined in cuda_path_tracer.cu after all includes.

struct GpuLightmapBakeTexel {
    Vec3 position;
    Vec3 normal;
    int texel_index;
};

__global__ void bake_lightmap_texels_kernel(
    const GpuScene* scene_ptr,
    const GpuLightmapBakeTexel* texels,
    int texel_count,
    int bake_samples,
    int bake_max_bounces,
    int sampling_mode,
    int mis_heuristic,
    int acceleration_structure,
    Vec3* lightmap_texels,
    uint8_t* lightmap_valid)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= texel_count) {
        return;
    }

    const GpuLightmapBakeTexel texel = texels[idx];
    const Vec3 position = texel.position;
    const Vec3 normal = texel.normal;

    RenderSettings bake_settings{};
    bake_settings.use_lightmap = false;
    bake_settings.use_irradiance_volume = false;
    bake_settings.stylized_samples = 0;
    bake_settings.stylized_max_depth = 0;
    bake_settings.max_bounces = bake_max_bounces;
    bake_settings.sampling_mode = static_cast<PathSamplingMode>(sampling_mode);
    bake_settings.mis_heuristic = static_cast<MisHeuristic>(mis_heuristic);
    bake_settings.acceleration_structure = static_cast<AccelerationStructure>(acceleration_structure);

    Vec3 front_irradiance{};
    Vec3 back_irradiance{};
    const Vec3 back_normal = mul(normal, -1.0f);

    for (int s = 0; s < bake_samples; ++s) {
        // Front hemisphere
        {
            uint32_t rng = hash_u32(
                static_cast<uint32_t>(idx) * 0x8da6b343u ^
                static_cast<uint32_t>(s) * 0xcb1ab31fu ^
                0x51ed270bu);
            const float u1 = rng_float(rng);
            const float u2 = rng_float(rng);
            const Vec3 local_dir = cosine_sample(u1, u2);
            const Vec3 direction = to_world_gpu(local_dir, normal);
            Ray ray{add(position, mul(normal, 0.002f)), direction};

            // Transparent skip + emissive check (matching CPU trace_lightmap_radiance)
            bool skip = false;
            Ray probe_ray = ray;
            for (int step = 0; step < 8; ++step) {
                GpuHit first_hit;
                if (!intersect_gpu(*scene_ptr, probe_ray, first_hit)) {
                    skip = true;
                    break;
                }
                if (first_hit.material < 0 || first_hit.material >= scene_ptr->material_count) {
                    skip = true;
                    break;
                }
                const GpuMaterial material = scene_ptr->materials[first_hit.material];
                uint32_t vis_rng2 = hash_u32(rng ^ static_cast<uint32_t>(step) * 0xd8163841u);
                if (!material_visible_gpu(*scene_ptr, material, first_hit.uv, vis_rng2)) {
                    probe_ray = {add(first_hit.position, mul(direction, 0.002f)), direction};
                    continue;
                }
                Vec3 emission;
                if (first_hit.triangle >= 0 && first_hit.triangle < scene_ptr->triangle_count) {
                    const GpuTriangle light = scene_ptr->triangles[first_hit.triangle];
                    emission = emitted_radiance_gpu(
                        light, material, light.emission,
                        material_emission_gpu(*scene_ptr, material, first_hit.uv),
                        light.light_double_sided != 0, direction);
                }
                if (has_light_emission_gpu(emission)) {
                    skip = true;
                }
                break;
            }
            if (!skip) {
                front_irradiance = add(front_irradiance,
                    mul(trace_gpu(*scene_ptr, ray, rng, bake_settings), kPi));
            }
        }

        // Back hemisphere
        {
            uint32_t rng = hash_u32(
                static_cast<uint32_t>(idx) * 0x8da6b343u ^
                static_cast<uint32_t>(s) * 0xcb1ab31fu ^
                0xa511e9b3u);
            const float u1 = rng_float(rng);
            const float u2 = rng_float(rng);
            const Vec3 local_dir = cosine_sample(u1, u2);
            const Vec3 direction = to_world_gpu(local_dir, back_normal);
            Ray ray{add(position, mul(back_normal, 0.002f)), direction};

            bool skip = false;
            Ray probe_ray = ray;
            for (int step = 0; step < 8; ++step) {
                GpuHit first_hit;
                if (!intersect_gpu(*scene_ptr, probe_ray, first_hit)) {
                    skip = true;
                    break;
                }
                if (first_hit.material < 0 || first_hit.material >= scene_ptr->material_count) {
                    skip = true;
                    break;
                }
                const GpuMaterial material = scene_ptr->materials[first_hit.material];
                uint32_t vis_rng2 = hash_u32(rng ^ static_cast<uint32_t>(step) * 0xd8163841u);
                if (!material_visible_gpu(*scene_ptr, material, first_hit.uv, vis_rng2)) {
                    probe_ray = {add(first_hit.position, mul(direction, 0.002f)), direction};
                    continue;
                }
                Vec3 emission;
                if (first_hit.triangle >= 0 && first_hit.triangle < scene_ptr->triangle_count) {
                    const GpuTriangle light = scene_ptr->triangles[first_hit.triangle];
                    emission = emitted_radiance_gpu(
                        light, material, light.emission,
                        material_emission_gpu(*scene_ptr, material, first_hit.uv),
                        light.light_double_sided != 0, direction);
                }
                if (has_light_emission_gpu(emission)) {
                    skip = true;
                }
                break;
            }
            if (!skip) {
                back_irradiance = add(back_irradiance,
                    mul(trace_gpu(*scene_ptr, ray, rng, bake_settings), kPi));
            }
        }
    }

    front_irradiance = divv(front_irradiance, static_cast<float>(bake_samples));
    back_irradiance = divv(back_irradiance, static_cast<float>(bake_samples));

    // Pick hemisphere with higher luminance (matching CPU).
    const float front_lum = front_irradiance.x * 0.2126f + front_irradiance.y * 0.7152f + front_irradiance.z * 0.0722f;
    const float back_lum = back_irradiance.x * 0.2126f + back_irradiance.y * 0.7152f + back_irradiance.z * 0.0722f;
    const Vec3 irradiance = back_lum > front_lum ? back_irradiance : front_irradiance;

    const int out_idx = texel.texel_index;
    lightmap_texels[out_idx] = {dclamp(irradiance.x, 0.0f, 1024.0f),
                                dclamp(irradiance.y, 0.0f, 1024.0f),
                                dclamp(irradiance.z, 0.0f, 1024.0f)};
    lightmap_valid[out_idx] = 1u;
}

__global__ void dilate_lightmap_kernel(
    Vec3* texels,
    uint8_t* valid,
    int width,
    int height,
    Vec3* next_texels,
    uint8_t* next_valid)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    if (idx >= total) {
        return;
    }

    // Copy existing valid texels directly.
    if (valid[idx]) {
        next_texels[idx] = texels[idx];
        next_valid[idx] = 1u;
        return;
    }

    const int x = idx % width;
    const int y = idx / width;

    Vec3 sum{};
    int count = 0;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            if (ox == 0 && oy == 0) {
                continue;
            }
            const int nx = x + ox;
            const int ny = y + oy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }
            const int neighbor = ny * width + nx;
            if (valid[neighbor]) {
                sum = add(sum, texels[neighbor]);
                ++count;
            }
        }
    }

    if (count > 0) {
        next_texels[idx] = divv(sum, static_cast<float>(count));
        next_valid[idx] = 1u;
    } else {
        next_texels[idx] = texels[idx];
        next_valid[idx] = 0u;
    }
}

// Rasterize lightmap triangles to compute per-texel world positions and normals.
// Returns the list of texels that fall inside a triangle.
inline std::vector<GpuLightmapBakeTexel> raster_lightmap_texels(
    const Lightmap& lightmap,
    const RenderScene& render_scene)
{
    std::vector<GpuLightmapBakeTexel> bake_texels;
    std::vector<int> owner(lightmap.texels.size(), -1);

    for (size_t tri_idx = 0; tri_idx < render_scene.triangles.size() && tri_idx < lightmap.triangles.size(); ++tri_idx) {
        const LightmapTriangle& lm_tri = lightmap.triangles[tri_idx];
        if (!lm_tri.valid) {
            continue;
        }
        const Vec2 p0{lm_tri.uv0.x * static_cast<float>(lightmap.width), lm_tri.uv0.y * static_cast<float>(lightmap.height)};
        const Vec2 p1{lm_tri.uv1.x * static_cast<float>(lightmap.width), lm_tri.uv1.y * static_cast<float>(lightmap.height)};
        const Vec2 p2{lm_tri.uv2.x * static_cast<float>(lightmap.width), lm_tri.uv2.y * static_cast<float>(lightmap.height)};
        const float area = edge_function(p0, p1, p2);
        if (std::fabs(area) <= 1.0e-8f) {
            continue;
        }
        const int min_x = std::clamp(static_cast<int>(std::floor(std::min({p0.x, p1.x, p2.x}))) - 1, 0, lightmap.width - 1);
        const int max_x = std::clamp(static_cast<int>(std::ceil(std::max({p0.x, p1.x, p2.x}))) + 1, 0, lightmap.width - 1);
        const int min_y = std::clamp(static_cast<int>(std::floor(std::min({p0.y, p1.y, p2.y}))) - 1, 0, lightmap.height - 1);
        const int max_y = std::clamp(static_cast<int>(std::ceil(std::max({p0.y, p1.y, p2.y}))) + 1, 0, lightmap.height - 1);
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                const Vec2 p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
                const float w0 = edge_function(p1, p2, p) / area;
                const float w1 = edge_function(p2, p0, p) / area;
                const float w2 = edge_function(p0, p1, p) / area;
                if (w0 >= -1.0e-4f && w1 >= -1.0e-4f && w2 >= -1.0e-4f) {
                    const int texel_idx = lightmap_texel_index(lightmap, x, y);
                    owner[static_cast<size_t>(texel_idx)] = static_cast<int>(tri_idx);
                }
            }
        }
    }

    for (size_t i = 0; i < owner.size(); ++i) {
        const int tri_idx = owner[i];
        if (tri_idx < 0) {
            continue;
        }
        const int x = static_cast<int>(i % static_cast<size_t>(lightmap.width));
        const int y = static_cast<int>(i / static_cast<size_t>(lightmap.width));
        const LightmapTriangle& lm_tri = lightmap.triangles[static_cast<size_t>(tri_idx)];
        const Triangle& tri = render_scene.triangles[static_cast<size_t>(tri_idx)];
        const Vec2 p0{lm_tri.uv0.x * static_cast<float>(lightmap.width), lm_tri.uv0.y * static_cast<float>(lightmap.height)};
        const Vec2 p1{lm_tri.uv1.x * static_cast<float>(lightmap.width), lm_tri.uv1.y * static_cast<float>(lightmap.height)};
        const Vec2 p2{lm_tri.uv2.x * static_cast<float>(lightmap.width), lm_tri.uv2.y * static_cast<float>(lightmap.height)};
        const float area = edge_function(p0, p1, p2);
        const Vec2 p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
        const float w0 = edge_function(p1, p2, p) / area;
        const float w1 = edge_function(p2, p0, p) / area;
        const float w2 = 1.0f - w0 - w1;
        const Vec3 position = tri.v0 * w0 + tri.v1 * w1 + tri.v2 * w2;
        Vec3 normal = normalize(tri.n0 * w0 + tri.n1 * w1 + tri.n2 * w2);
        if (dot(normal, normal) <= 0.0f) {
            normal = tri.normal;
        }
        GpuLightmapBakeTexel bt;
        bt.position = position;
        bt.normal = normal;
        bt.texel_index = static_cast<int>(i);
        bake_texels.push_back(bt);
    }
    return bake_texels;
}

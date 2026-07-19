// GPU irradiance volume bake kernels.
// Included from cuda_path_tracer.cu inside the anonymous namespace.
// The host orchestrator is defined in cuda_path_tracer.cu after all includes.

__global__ void bake_radiance_kernel(
    const GpuScene* scene_ptr,
    const Vec3* probe_positions,
    int probe_count,
    const Vec3* directions,
    int direction_count,
    int bake_samples,
    int bake_max_bounces,
    int sampling_mode,
    int mis_heuristic,
    int acceleration_structure,
    float emissive_intensity_scale,
    Vec3* probe_radiance)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = probe_count * direction_count;
    if (idx >= total) {
        return;
    }

    const int probe_idx = idx / direction_count;
    const int dir_idx = idx % direction_count;

    const Vec3 origin = probe_positions[probe_idx];
    const Vec3 direction = directions[dir_idx];

    RenderSettings bake_settings{};
    bake_settings.use_irradiance_volume = false;
    bake_settings.stylized_samples = 0;
    bake_settings.stylized_max_depth = 0;
    bake_settings.max_bounces = bake_max_bounces;
    bake_settings.sampling_mode = static_cast<PathSamplingMode>(sampling_mode);
    bake_settings.mis_heuristic = static_cast<MisHeuristic>(mis_heuristic);
    bake_settings.acceleration_structure = static_cast<AccelerationStructure>(acceleration_structure);
    bake_settings.emissive_intensity_scale = emissive_intensity_scale;

    // Probe forward through transparent surfaces (matching CPU trace_volume_radiance).
    Ray probe_ray{add(origin, mul(direction, 0.002f)), direction};
    for (int step = 0; step < 8; ++step) {
        GpuHit first_hit;
        if (!intersect_gpu(*scene_ptr, probe_ray, first_hit)) {
            probe_radiance[idx] = Vec3{};
            return;
        }
        if (first_hit.material < 0 || first_hit.material >= scene_ptr->material_count) {
            probe_radiance[idx] = Vec3{};
            return;
        }
        const GpuMaterial material = scene_ptr->materials[first_hit.material];
        uint32_t vis_rng = hash_u32(
            static_cast<uint32_t>(idx) * 0x8da6b343u ^
            static_cast<uint32_t>(step) * 0xd8163841u ^
            0x9e3779b9u);
        if (!material_visible_gpu(*scene_ptr, material, first_hit.uv, vis_rng)) {
            probe_ray = {add(first_hit.position, mul(direction, 0.002f)), direction};
            continue;
        }
        // If we hit an emissive surface directly, return zero radiance (matching CPU).
        Vec3 emission;
        if (first_hit.triangle >= 0 && first_hit.triangle < scene_ptr->triangle_count) {
            const GpuTriangle light = scene_ptr->triangles[first_hit.triangle];
            emission = emitted_radiance_gpu(
                light,
                material,
                light.emission,
                material_emission_gpu(*scene_ptr, material, first_hit.uv, bake_settings),
                light.light_double_sided != 0,
                direction);
        }
        if (has_light_emission_gpu(emission)) {
            probe_radiance[idx] = Vec3{};
            return;
        }
        break;
    }

    // Trace bake_samples paths and average.
    Vec3 estimate{};
    for (int s = 0; s < bake_samples; ++s) {
        uint32_t rng = hash_u32(
            static_cast<uint32_t>(idx) * 0x8da6b343u ^
            static_cast<uint32_t>(s) * 0xcb1ab31fu ^
            0x9e3779b9u);
        estimate = add(estimate, trace_gpu(*scene_ptr, probe_ray, rng, bake_settings));
    }
    probe_radiance[idx] = divv(estimate, static_cast<float>(bake_samples));
}

__global__ void reduce_irradiance_kernel(
    const Vec3* probe_radiance,
    int probe_count,
    int direction_count,
    const float* cosine_weights,
    Vec3* probe_irradiance,
    float clamp_limit)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = probe_count * direction_count;
    if (idx >= total) {
        return;
    }

    const int probe_idx = idx / direction_count;
    const int normal_idx = idx % direction_count;

    Vec3 irradiance{};
    const int weight_offset = normal_idx * direction_count;
    const int radiance_offset = probe_idx * direction_count;
    for (int j = 0; j < direction_count; ++j) {
        const float w = cosine_weights[weight_offset + j];
        irradiance = add(irradiance, mul(probe_radiance[radiance_offset + j], w));
    }

    irradiance.x = dclamp(irradiance.x, 0.0f, clamp_limit);
    irradiance.y = dclamp(irradiance.y, 0.0f, clamp_limit);
    irradiance.z = dclamp(irradiance.z, 0.0f, clamp_limit);

    probe_irradiance[idx] = irradiance;
}

// Collect all probe positions from the sparse grid into a flat array.
inline void collect_probe_positions(const IrradianceVolumeGrid& grid, std::vector<Vec3>& positions) {
    for (const IrradianceVolumeSample& sample : grid.samples) {
        positions.push_back(sample.position);
    }
    for (const IrradianceVolumeCell& cell : grid.cells) {
        if (cell.subgrid) {
            collect_probe_positions(*cell.subgrid, positions);
        }
    }
}

// Distribute irradiance results back into the IrradianceVolume grid.
inline void populate_irradiance_results(
    IrradianceVolumeGrid& grid,
    const std::vector<Vec3>& irradiance_results,
    int direction_count,
    int& sample_idx)
{
    for (IrradianceVolumeSample& sample : grid.samples) {
        sample.irradiance.resize(static_cast<size_t>(direction_count));
        for (int d = 0; d < direction_count; ++d) {
            sample.irradiance[static_cast<size_t>(d)] =
                irradiance_results[static_cast<size_t>(sample_idx * direction_count + d)];
        }
        ++sample_idx;
    }
    for (IrradianceVolumeCell& cell : grid.cells) {
        if (cell.subgrid) {
            populate_irradiance_results(*cell.subgrid, irradiance_results, direction_count, sample_idx);
        }
    }
}

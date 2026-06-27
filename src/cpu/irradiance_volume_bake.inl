void expand_irradiance_bounds(Aabb& bounds, Vec3 point) {
    bounds.min = min(bounds.min, point);
    bounds.max = max(bounds.max, point);
}

Aabb triangle_irradiance_bounds(const Triangle& triangle) {
    Aabb bounds;
    expand_irradiance_bounds(bounds, triangle.v0);
    expand_irradiance_bounds(bounds, triangle.v1);
    expand_irradiance_bounds(bounds, triangle.v2);
    return bounds;
}

Aabb sphere_irradiance_bounds(const RenderSphere& sphere) {
    const Vec3 r{sphere.radius};
    return {sphere.center - r, sphere.center + r};
}

Aabb scene_irradiance_bounds(const RenderScene& render_scene, const RenderSettings& settings) {
    Aabb bounds;
    if (settings.irradiance_volume_manual_bounds) {
        bounds.min = min(settings.irradiance_volume_bounds_min, settings.irradiance_volume_bounds_max);
        bounds.max = max(settings.irradiance_volume_bounds_min, settings.irradiance_volume_bounds_max);
    } else {
        for (const Triangle& triangle : render_scene.triangles) {
            expand_irradiance_bounds(bounds, triangle.v0);
            expand_irradiance_bounds(bounds, triangle.v1);
            expand_irradiance_bounds(bounds, triangle.v2);
        }
        for (const RenderSphere& sphere : render_scene.spheres) {
            const Vec3 r{sphere.radius};
            expand_irradiance_bounds(bounds, sphere.center - r);
            expand_irradiance_bounds(bounds, sphere.center + r);
        }
    }

    if (!aabb_is_valid(bounds)) {
        bounds.min = {-1.0f, -1.0f, -1.0f};
        bounds.max = {1.0f, 1.0f, 1.0f};
        return bounds;
    }

    const Vec3 extent = bounds.max - bounds.min;
    const float max_extent = std::max({extent.x, extent.y, extent.z, 1.0e-3f});
    for (int axis = 0; axis < 3; ++axis) {
        if (axis_component(extent, axis) <= 1.0e-5f) {
            const float center = (axis_component(bounds.min, axis) + axis_component(bounds.max, axis)) * 0.5f;
            bounds.min = with_axis(bounds.min, axis, center - max_extent * 0.5f);
            bounds.max = with_axis(bounds.max, axis, center + max_extent * 0.5f);
        }
    }

    const float inset = std::clamp(settings.irradiance_volume_bounds_inset, 0.0f, 0.45f);
    if (inset > 0.0f) {
        const Aabb original = bounds;
        const Vec3 inset_amount = (bounds.max - bounds.min) * inset;
        bounds.min += inset_amount;
        bounds.max = bounds.max - inset_amount;
        if (!aabb_is_valid(bounds)) {
            bounds = original;
        }
    }
    return bounds;
}

bool cell_contains_irradiance_geometry(const RenderScene& render_scene, const Aabb& cell_bounds) {
    for (const Triangle& triangle : render_scene.triangles) {
        if (aabb_overlaps(cell_bounds, triangle_irradiance_bounds(triangle))) {
            return true;
        }
    }
    for (const RenderSphere& sphere : render_scene.spheres) {
        if (aabb_overlaps(cell_bounds, sphere_irradiance_bounds(sphere))) {
            return true;
        }
    }
    return false;
}

void initialize_irradiance_grid(
    IrradianceVolume& volume,
    IrradianceVolumeGrid& grid,
    const RenderScene& render_scene,
    const Aabb& bounds,
    int resolution,
    bool build_subgrids) {
    resolution = std::max(2, resolution);
    grid.bounds = bounds;
    grid.resolution = resolution;
    grid.samples.clear();
    grid.cells.clear();
    const size_t sample_count = static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * static_cast<size_t>(resolution);
    grid.samples.resize(sample_count);
    for (int z = 0; z < resolution; ++z) {
        for (int y = 0; y < resolution; ++y) {
            for (int x = 0; x < resolution; ++x) {
                IrradianceVolumeSample& sample = grid.samples[static_cast<size_t>(grid_sample_index(resolution, x, y, z))];
                sample.position = grid_position(bounds, resolution, x, y, z);
                sample.irradiance.assign(volume.directions.size(), Vec3{});
            }
        }
    }
    volume.spatial_sample_count += sample_count;

    if (!build_subgrids) {
        return;
    }

    const int cells_per_axis = resolution - 1;
    const size_t cell_count = static_cast<size_t>(cells_per_axis) * static_cast<size_t>(cells_per_axis) * static_cast<size_t>(cells_per_axis);
    grid.cells.resize(cell_count);
    volume.first_level_cell_count = cell_count;
    for (int z = 0; z < cells_per_axis; ++z) {
        for (int y = 0; y < cells_per_axis; ++y) {
            for (int x = 0; x < cells_per_axis; ++x) {
                const Aabb cell_bounds = grid_cell_bounds(bounds, resolution, x, y, z);
                if (!cell_contains_irradiance_geometry(render_scene, cell_bounds)) {
                    continue;
                }
                IrradianceVolumeCell& cell = grid.cells[static_cast<size_t>(grid_cell_index(resolution, x, y, z))];
                cell.subgrid = std::make_unique<IrradianceVolumeGrid>();
                ++volume.subgrid_count;
                initialize_irradiance_grid(
                    volume,
                    *cell.subgrid,
                    render_scene,
                    cell_bounds,
                    volume.subgrid_resolution,
                    false);
            }
        }
    }
}

Vec3 trace_volume_radiance(
    const RenderScene& render_scene,
    const Scene& scene,
    Vec3 origin,
    Vec3 direction,
    Rng& rng,
    RenderSettings bake_settings) {
    bake_settings.use_irradiance_volume = false;
    bake_settings.stylized_samples = 0;
    bake_settings.stylized_max_depth = 0;
    bake_settings.max_bounces = std::max(1, bake_settings.irradiance_volume_bake_bounces);

    Ray probe_ray{origin + direction * 0.002f, direction};
    Rng visibility_rng = rng;
    for (int step = 0; step < 8; ++step) {
        Hit first_hit;
        if (!intersect_scene(render_scene, probe_ray, first_hit, bake_settings.acceleration_structure)) {
            return {};
        }
        if (first_hit.material < 0 || first_hit.material >= static_cast<int>(scene.materials.size())) {
            return {};
        }
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(first_hit.material)];
        if (!material) {
            return {};
        }
        if (!material_visible(*material, first_hit.uv, visibility_rng)) {
            probe_ray = {first_hit.position + direction * 0.002f, direction};
            continue;
        }
        Vec3 emission;
        if (first_hit.triangle >= 0 && first_hit.triangle < static_cast<int>(render_scene.triangles.size())) {
            emission = emitted_radiance(scene, render_scene.triangles[static_cast<size_t>(first_hit.triangle)], first_hit.uv, direction);
        }
        if (has_light_emission(emission)) {
            return {};
        }
        break;
    }
    return trace_path(render_scene, scene, probe_ray, rng, bake_settings, nullptr);
}

void bake_irradiance_sample(
    IrradianceVolume& volume,
    IrradianceVolumeSample& sample,
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings,
    uint32_t sample_serial) {
    const size_t direction_count = volume.directions.size();
    sample.irradiance.assign(direction_count, Vec3{});
    if (direction_count == 0) {
        return;
    }

    std::vector<Vec3> radiance(direction_count, Vec3{});
    const int bake_samples = std::max(1, volume.bake_samples);
    for (size_t direction_index = 0; direction_index < direction_count; ++direction_index) {
        const Vec3 direction = volume.directions[direction_index];
        Vec3 estimate;
        for (int sample_index = 0; sample_index < bake_samples; ++sample_index) {
            const uint32_t seed = hash_u32(
                sample_serial * 0x8da6b343u ^
                static_cast<uint32_t>(direction_index) * 0xd8163841u ^
                static_cast<uint32_t>(sample_index) * 0xcb1ab31fu ^
                0x9e3779b9u);
            Rng rng(seed);
            estimate += trace_volume_radiance(render_scene, scene, sample.position, direction, rng, settings);
            ++volume.radiance_trace_count;
        }
        radiance[direction_index] = estimate / static_cast<float>(bake_samples);
    }

    for (size_t normal_index = 0; normal_index < direction_count; ++normal_index) {
        Vec3 irradiance;
        const size_t weight_offset = normal_index * direction_count;
        for (size_t radiance_index = 0; radiance_index < direction_count; ++radiance_index) {
            irradiance += radiance[radiance_index] * volume.cosine_weights[weight_offset + radiance_index];
        }
        sample.irradiance[normal_index] = clamp_sample_radiance(irradiance, 1024.0f);
    }
}

void bake_irradiance_grid(
    IrradianceVolume& volume,
    IrradianceVolumeGrid& grid,
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings,
    uint32_t& sample_serial) {
    for (IrradianceVolumeSample& sample : grid.samples) {
        bake_irradiance_sample(volume, sample, render_scene, scene, settings, sample_serial++);
    }
    for (IrradianceVolumeCell& cell : grid.cells) {
        if (cell.subgrid) {
            bake_irradiance_grid(volume, *cell.subgrid, render_scene, scene, settings, sample_serial);
        }
    }
}

std::shared_ptr<IrradianceVolume> build_irradiance_volume(
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings) {
    const auto begin = std::chrono::steady_clock::now();
    auto volume = std::make_shared<IrradianceVolume>();
    volume->grid_resolution = std::max(2, settings.irradiance_volume_grid_resolution);
    volume->subgrid_resolution = std::max(2, settings.irradiance_volume_subgrid_resolution);
    volume->direction_resolution = std::max(1, settings.irradiance_volume_direction_resolution);
    volume->bake_samples = std::max(1, settings.irradiance_volume_bake_samples);
    volume->bake_bounces = std::max(1, settings.irradiance_volume_bake_bounces);
    volume->bounds = scene_irradiance_bounds(render_scene, settings);
    volume->directions = make_irradiance_volume_directions(volume->direction_resolution);
    volume->cosine_weights = make_irradiance_volume_weights(volume->directions);

    initialize_irradiance_grid(*volume, volume->grid, render_scene, volume->bounds, volume->grid_resolution, true);
    collect_debug_probes_from_grid(*volume, volume->grid);
    volume->unique_debug_probe_count = volume->debug_probes.size();

    uint32_t sample_serial = 1u;
    bake_irradiance_grid(*volume, volume->grid, render_scene, scene, settings, sample_serial);

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const size_t irradiance_bytes = volume->spatial_sample_count * volume->directions.size() * sizeof(Vec3);
    const size_t weight_bytes = volume->cosine_weights.size() * sizeof(float);
    LT_LOG_INFO(
        "Irradiance volume baked: samples={} debug_probes={} first_cells={} subgrids={} directions={} rays={} memory_kib={} elapsed_ms={}",
        volume->spatial_sample_count,
        volume->unique_debug_probe_count,
        volume->first_level_cell_count,
        volume->subgrid_count,
        volume->directions.size(),
        volume->radiance_trace_count,
        (irradiance_bytes + weight_bytes) / 1024u,
        elapsed_ms);
    return volume;
}

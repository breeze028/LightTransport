void CpuPathTracer::reset() {
    cached_render_scene_ = {};
    cached_irradiance_volume_.reset();
    cached_lightmap_.reset();
    scene_uploaded_ = false;
}

Ray make_center_camera_ray(const Camera& camera, int x, int y, const RenderSettings& settings) {
    const float aspect = static_cast<float>(settings.width) / static_cast<float>(settings.height);
    const float half_height = std::tan(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = normalize(camera.target - camera.position);
    const float right_sign = camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const Vec3 right = normalize(cross(forward, camera.up)) * right_sign;
    const Vec3 up = cross(right, forward) * right_sign;
    const float sample_x = 0.5f + settings.camera_jitter_x;
    const float sample_y = 0.5f + settings.camera_jitter_y;
    const float u = ((static_cast<float>(x) + sample_x) / static_cast<float>(settings.width) * 2.0f - 1.0f) * half_width;
    const float v = (1.0f - (static_cast<float>(y) + sample_y) / static_cast<float>(settings.height) * 2.0f) * half_height;
    return {camera.position, normalize(forward + right * u + up * v)};
}

uint32_t primary_object_id(const Hit& hit) {
    if (hit.mesh >= 0) {
        return static_cast<uint32_t>(hit.mesh + 1);
    }
    if (hit.triangle >= 0) {
        return 0x40000000u | static_cast<uint32_t>(hit.triangle + 1);
    }
    if (hit.material >= 0) {
        return 0x80000000u | static_cast<uint32_t>(hit.material + 1);
    }
    return 0u;
}

void write_primary_aov(const RenderScene& render_scene,
                       const Scene& scene,
                       const RenderSettings& settings,
                       int x,
                       int y,
                       size_t idx,
                       Framebuffer& framebuffer) {
    Framebuffer::AovBuffers& aov = framebuffer.aov;
    aov.albedo[idx] = {1.0f, 1.0f, 1.0f};
    aov.emission[idx] = {};
    aov.normal[idx] = {};
    aov.world_position[idx] = {};
    aov.linear_depth[idx] = kInfinity;
    aov.object_id[idx] = 0u;

    Ray ray = make_center_camera_ray(scene.camera, x, y, settings);
    Rng visibility_rng(make_pixel_seed(static_cast<uint32_t>(x), static_cast<uint32_t>(y), settings.frame_index ^ 0x9e3779b9u));
    for (int transparent_step = 0; transparent_step < 8; ++transparent_step) {
        Hit hit;
        if (!intersect_scene(render_scene, ray, hit, settings.acceleration_structure)) {
            return;
        }
        if (hit.material < 0 || hit.material >= static_cast<int>(scene.materials.size())) {
            return;
        }
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(hit.material)];
        if (!material) {
            return;
        }
        if (!material_visible(*material, hit.uv, visibility_rng)) {
            ray = {hit.position + ray.direction * 0.002f, ray.direction};
            continue;
        }

        hit.normal = apply_normal_map(*material, hit, ray.direction);
        Vec3 emission = material->emitted(hit.uv);
        if (hit.triangle >= 0 && hit.triangle < static_cast<int>(render_scene.triangles.size())) {
            emission = emitted_radiance(scene, render_scene.triangles[static_cast<size_t>(hit.triangle)], hit.uv, ray.direction, settings);
        }
        const Vec3 forward = normalize(scene.camera.target - scene.camera.position);
        aov.albedo[idx] = max(material->base_color(hit.uv), Vec3{0.001f, 0.001f, 0.001f});
        aov.emission[idx] = max(emission, Vec3{});
        aov.normal[idx] = hit.normal;
        aov.world_position[idx] = hit.position;
        aov.linear_depth[idx] = std::max(0.0f, dot(hit.position - scene.camera.position, forward));
        aov.object_id[idx] = primary_object_id(hit);
        return;
    }
}

void CpuPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    if (!scene_uploaded_ ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform)) {
        LT_LOG_DEBUG(
            "CPU render scene rebuild: meshes={} spheres={} materials={} textures={} accel={}",
            scene.meshes.size(),
            scene.spheres.size(),
            scene.materials.size(),
            scene.textures.size(),
            static_cast<int>(settings.acceleration_structure));
        cached_render_scene_ = build_render_scene(scene);
        LT_LOG_DEBUG(
            "CPU render scene built: triangles={} light_triangles={}",
            cached_render_scene_.triangles.size(),
            cached_render_scene_.light_triangle_indices.size());
        scene_uploaded_ = true;
    }
    const RenderScene& render_scene = cached_render_scene_;
    std::shared_ptr<Lightmap> lightmap;
    if (lightmap_rendering_enabled(settings)) {
        const bool lightmap_dirty = has_dirty(settings.dirty, RenderDirty::Lightmap) ||
            has_dirty(settings.dirty, RenderDirty::Geometry) ||
            has_dirty(settings.dirty, RenderDirty::Transform) ||
            has_dirty(settings.dirty, RenderDirty::Material) ||
            has_dirty(settings.dirty, RenderDirty::Texture) ||
            has_dirty(settings.dirty, RenderDirty::Environment);
        bool lightmap_rebuilt = false;
        lightmap = update_lightmap(
            cached_lightmap_,
            cached_render_scene_,
            scene,
            settings,
            lightmap_dirty,
            lightmap_rebuilt);
    } else if (cached_lightmap_) {
        cached_lightmap_.reset();
        apply_lightmap_to_render_scene(nullptr, cached_render_scene_);
        set_lightmap_progress_phase(settings, LightmapBakePhase::Idle);
    }
    std::shared_ptr<IrradianceVolume> irradiance_volume;
    if (irradiance_volume_rendering_enabled(settings)) {
        const bool volume_dirty = has_dirty(settings.dirty, RenderDirty::IrradianceVolume) ||
            has_dirty(settings.dirty, RenderDirty::Geometry) ||
            has_dirty(settings.dirty, RenderDirty::Material) ||
            has_dirty(settings.dirty, RenderDirty::Texture) ||
            has_dirty(settings.dirty, RenderDirty::Environment);
        bool volume_rebuilt = false;
        irradiance_volume = update_irradiance_volume(
            cached_irradiance_volume_,
            render_scene,
            scene,
            settings,
            volume_dirty,
            volume_rebuilt);
    } else if (cached_irradiance_volume_) {
        cached_irradiance_volume_.reset();
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Idle);
    }

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const unsigned int thread_count = std::max(1u, hardware_threads > 1u ? hardware_threads - 1u : hardware_threads);
    std::atomic<int> next_row{0};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    const bool use_rasterized_aov = svgf_denoising_enabled(settings) && has_rasterized_svgf_aov(framebuffer);
    if (svgf_denoising_enabled(settings) && !use_rasterized_aov) {
        framebuffer.aov.clear();
    }

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
                    sample += trace_path_with_irradiance_probe_debug(
                        render_scene,
                        scene,
                        make_camera_ray(scene.camera, x, y, settings, rng),
                        rng,
                        settings,
                        irradiance_volume.get(),
                        lightmap.get());
                }
                sample = sample / static_cast<float>(settings.samples_per_pixel);
                if (svgf_denoising_enabled(settings)) {
                    framebuffer.aov.radiance[idx] = sample;
                    if (!use_rasterized_aov) {
                        write_primary_aov(render_scene, scene, settings, x, y, idx, framebuffer);
                    }
                    framebuffer.rgba[idx] = to_rgba8(sample);
                } else {
                    framebuffer.accumulation[idx] += sample;
                    framebuffer.rgba[idx] = to_rgba8(framebuffer.accumulation[idx] / static_cast<float>(settings.frame_index + 1u));
                }
            }
        }
    };

    for (unsigned int i = 0; i < thread_count; ++i) {
        workers.emplace_back(render_rows);
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    if (svgf_denoising_enabled(settings)) {
        apply_svgf_denoiser(scene, settings, framebuffer);
    }
}

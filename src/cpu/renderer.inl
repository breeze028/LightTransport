void CpuPathTracer::reset() {
    cached_render_scene_ = {};
    cached_irradiance_volume_.reset();
    scene_uploaded_ = false;
}

void CpuPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    if (!scene_uploaded_ || has_dirty(settings.dirty, RenderDirty::Geometry)) {
        LT_LOG_DEBUG(
            "CPU render scene rebuild: meshes={} spheres={} materials={} textures={} accel={}",
            scene.meshes.size(),
            scene.spheres.size(),
            scene.materials.size(),
            scene.textures.size(),
            static_cast<int>(settings.acceleration_structure));
        cached_render_scene_ = build_render_scene(scene);
        scene_uploaded_ = true;
    }
    const RenderScene& render_scene = cached_render_scene_;
    std::shared_ptr<IrradianceVolume> irradiance_volume;
    if (irradiance_volume_rendering_enabled(settings)) {
        const bool volume_dirty = has_dirty(settings.dirty, RenderDirty::IrradianceVolume) ||
            has_dirty(settings.dirty, RenderDirty::Geometry) ||
            has_dirty(settings.dirty, RenderDirty::Material) ||
            has_dirty(settings.dirty, RenderDirty::Texture) ||
            has_dirty(settings.dirty, RenderDirty::Environment);
        if (!cached_irradiance_volume_ || volume_dirty) {
            cached_irradiance_volume_ = build_irradiance_volume(render_scene, scene, settings);
        }
        irradiance_volume = std::static_pointer_cast<IrradianceVolume>(cached_irradiance_volume_);
    } else if (cached_irradiance_volume_) {
        cached_irradiance_volume_.reset();
    }

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
                    sample += trace_path_with_irradiance_probe_debug(
                        render_scene,
                        scene,
                        make_camera_ray(scene.camera, x, y, settings, rng),
                        rng,
                        settings,
                        irradiance_volume.get());
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

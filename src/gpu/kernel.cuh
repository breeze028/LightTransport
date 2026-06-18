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
        sample = add(sample, trace_gpu(scene, camera_ray(scene.camera, x, y, settings.width, settings.height, rng), rng, settings));
    }
    sample = divv(sample, static_cast<float>(settings.samples_per_pixel));
    accumulation[idx] = add(accumulation[idx], sample);
    rgba[idx] = rgba8_gpu(divv(accumulation[idx], static_cast<float>(settings.frame_index + 1u)));
}

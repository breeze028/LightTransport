#pragma once

#include "lt/scene.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lt {

struct RenderSettings {
    int width = 1280;
    int height = 720;
    int samples_per_pixel = 1;
    int max_bounces = 6;
    uint32_t frame_index = 0;
};

struct Framebuffer {
    int width = 0;
    int height = 0;
    std::vector<Vec3> accumulation;
    std::vector<uint32_t> rgba;

    void resize(int w, int h) {
        if (w == width && h == height) {
            return;
        }
        width = w;
        height = h;
        accumulation.assign(static_cast<size_t>(w) * static_cast<size_t>(h), Vec3{});
        rgba.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0xff000000u);
    }

    void clear() {
        std::fill(accumulation.begin(), accumulation.end(), Vec3{});
        std::fill(rgba.begin(), rgba.end(), 0xff000000u);
    }
};

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual const char* name() const = 0;
    virtual bool available() const = 0;
    virtual void reset() = 0;
    virtual void render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) = 0;
};

class CpuPathTracer final : public IRenderer {
public:
    const char* name() const override { return "CPU Path Tracer"; }
    bool available() const override { return true; }
    void reset() override;
    void render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) override;

private:
    RenderScene cached_render_scene_;
    uint64_t cached_scene_signature_ = 0;
};

class CudaPathTracer final : public IRenderer {
public:
    ~CudaPathTracer() override;
    const char* name() const override { return "CUDA Path Tracer"; }
    bool available() const override;
    void reset() override;
    void render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) override;

private:
    void* device_accumulation_ = nullptr;
    void* device_rgba_ = nullptr;
    void* device_scene_ = nullptr;
    size_t cached_pixels_ = 0;
};

} // namespace lt

#pragma once

#include "lt/scene.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lt {

enum class MisHeuristic {
    Balance = 0,
    Power = 1,
};

enum class AccelerationStructure {
    Auto = 0,
    Flat = 1,
    TwoLevel = 2,
};

enum class RenderDirty : uint32_t {
    None = 0,
    Render = 1u << 0u,
    Camera = 1u << 1u,
    Material = 1u << 2u,
    Texture = 1u << 3u,
    Geometry = 1u << 4u,
    Environment = 1u << 5u,
    All = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u) | (1u << 4u) | (1u << 5u),
};

constexpr RenderDirty operator|(RenderDirty a, RenderDirty b) {
    return static_cast<RenderDirty>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

constexpr RenderDirty operator&(RenderDirty a, RenderDirty b) {
    return static_cast<RenderDirty>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

constexpr bool has_dirty(RenderDirty value, RenderDirty flag) {
    return static_cast<uint32_t>(value & flag) != 0u;
}

struct RenderSettings {
    int width = 1280;
    int height = 720;
    int samples_per_pixel = 1;
    int max_bounces = 6;
    bool use_mis = false;
    MisHeuristic mis_heuristic = MisHeuristic::Power;
    AccelerationStructure acceleration_structure = AccelerationStructure::Auto;
    int stylized_samples = 8;
    int stylized_max_depth = 1;
    uint32_t frame_index = 0;
    RenderDirty dirty = RenderDirty::All;
};

inline bool scene_has_npr_styles(const Scene& scene) {
    for (const std::shared_ptr<Material>& material : scene.materials) {
        if (material && material->npr.style != NprStyle::None) {
            return true;
        }
    }
    return false;
}

inline bool stylized_rendering_enabled(const RenderSettings& settings, const Scene& scene) {
    return settings.stylized_samples > 0 && settings.stylized_max_depth > 0 && scene_has_npr_styles(scene);
}

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
    bool scene_uploaded_ = false;
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
    void* device_materials_ = nullptr;
    void* device_textures_ = nullptr;
    void* device_triangles_ = nullptr;
    void* device_spheres_ = nullptr;
    void* device_triangle_indices_ = nullptr;
    void* device_light_indices_ = nullptr;
    void* device_bvh_nodes_ = nullptr;
    void* device_mesh_instances_ = nullptr;
    void* device_mesh_instance_indices_ = nullptr;
    void* device_tlas_nodes_ = nullptr;
    std::vector<void*> texture_arrays_;
    std::vector<uint64_t> texture_objects_;
    size_t cached_pixels_ = 0;
    int cached_materials_ = 0;
    int cached_textures_ = 0;
    int cached_triangles_ = 0;
    int cached_spheres_ = 0;
    int cached_triangle_indices_ = 0;
    int cached_lights_ = 0;
    int cached_bvh_nodes_ = 0;
    int cached_mesh_instances_ = 0;
    int cached_mesh_instance_indices_ = 0;
    int cached_tlas_nodes_ = 0;
    bool scene_uploaded_ = false;
};

} // namespace lt

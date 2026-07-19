#pragma once

#include "lt/scene.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace lt {

enum class MisHeuristic {
    Balance = 0,
    Power = 1,
};

enum class PathSamplingMode {
    Unidirectional = 0,
    NextEventEstimation = 1,
    MultipleImportanceSampling = 2,
};

enum class RestirBiasCorrection {
    Basic = 0,
    RayTraced = 1,
};

enum class RestirGiResamplingMode {
    None = 0,
    Temporal = 1,
    TemporalSpatial = 2,
};

enum class RestirPtResamplingMode {
    None = 0,
    Temporal = 1,
    TemporalSpatial = 2,
};

enum class RestirPtReconnectionMode {
    FixedThreshold = 0,
    Footprint = 1,
};

enum class AccelerationStructure {
    Flat = 0,
    TwoLevel = 1,
};

enum class IrradianceVolumeBakePhase : int {
    Idle = 0,
    LoadingCache = 1,
    Baking = 2,
    SavingCache = 3,
    Complete = 4,
    Failed = 5,
};

enum class IrradianceVolumeBakeBackend : int {
    Gpu = 0,
    Cpu = 1,
};

enum class LightmapBakeBackend : int {
    Gpu = 0,
    Cpu = 1,
};

enum class DenoiserMode : int {
    Off = 0,
    Svgf = 1,
};

enum class DenoiserDebugView : int {
    Final = 0,
    Raw = 1,
    Albedo = 2,
    Normal = 3,
    Depth = 4,
    Variance = 5,
    HistoryLength = 6,
};

enum class AntialiasingMode : int {
    Off = 0,
    StablePostAA = 1,
    TAA = 2,
};

enum class LightmapBakePhase : int {
    Idle = 0,
    LoadingCache = 1,
    Unwrapping = 2,
    Baking = 3,
    SavingCache = 4,
    Complete = 5,
    Failed = 6,
};

struct IrradianceVolumeBakeProgress {
    std::atomic<int> phase{static_cast<int>(IrradianceVolumeBakePhase::Idle)};
    std::atomic<uint64_t> total_samples{0};
    std::atomic<uint64_t> completed_samples{0};
    std::atomic<uint64_t> total_rays{0};
    std::atomic<uint64_t> traced_rays{0};
    std::atomic<int> direction_count{0};
    std::atomic<double> elapsed_ms{0.0};
};

struct LightmapBakeProgress {
    std::atomic<int> phase{static_cast<int>(LightmapBakePhase::Idle)};
    std::atomic<uint64_t> total_texels{0};
    std::atomic<uint64_t> completed_texels{0};
    std::atomic<uint64_t> total_rays{0};
    std::atomic<uint64_t> traced_rays{0};
    std::atomic<int> width{0};
    std::atomic<int> height{0};
    std::atomic<double> elapsed_ms{0.0};
};

enum class RenderDirty : uint32_t {
    None = 0,
    Render = 1u << 0u,
    Camera = 1u << 1u,
    Material = 1u << 2u,
    Texture = 1u << 3u,
    Geometry = 1u << 4u,
    Environment = 1u << 5u,
    IrradianceVolume = 1u << 6u,
    Transform = 1u << 7u,
    Lightmap = 1u << 8u,
    All = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u) | (1u << 4u) | (1u << 5u) | (1u << 6u) | (1u << 7u) | (1u << 8u),
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
    PathSamplingMode sampling_mode = PathSamplingMode::NextEventEstimation;
    MisHeuristic mis_heuristic = MisHeuristic::Power;
    AccelerationStructure acceleration_structure = AccelerationStructure::TwoLevel;
    float emissive_intensity_scale = 1.0f;
    int stylized_samples = 8;
    int stylized_max_depth = 1;
    bool use_irradiance_volume = false;
    int irradiance_volume_grid_resolution = 7;
    int irradiance_volume_subgrid_resolution = 3;
    int irradiance_volume_direction_resolution = 9;
    int irradiance_volume_bake_samples = 1;
    int irradiance_volume_bake_bounces = 4;
    float irradiance_volume_bounds_inset = 0.01f;
    IrradianceVolumeBakeBackend irradiance_volume_bake_backend = IrradianceVolumeBakeBackend::Gpu;
    bool irradiance_volume_principled_gi = false;
    bool irradiance_volume_debug_probes = false;
    float irradiance_volume_debug_probe_radius_scale = 0.10f;
    bool irradiance_volume_cache_enabled = true;
    bool irradiance_volume_auto_update = true;
    bool irradiance_volume_force_rebake = false;
    char irradiance_volume_cache_path[1024] = {};
    char irradiance_volume_cache_key[1024] = {};
    IrradianceVolumeBakeProgress* irradiance_volume_bake_progress = nullptr;
    bool irradiance_volume_manual_bounds = false;
    Vec3 irradiance_volume_bounds_min = {-1.0f, -1.0f, -1.0f};
    Vec3 irradiance_volume_bounds_max = {1.0f, 1.0f, 1.0f};
    bool use_lightmap = false;
    int lightmap_resolution = 1024;
    int lightmap_padding = 2;
    int lightmap_dilation = 4;
    int lightmap_bake_samples = 4;
    int lightmap_bake_bounces = 4;
    bool lightmap_principled_gi = false;
    LightmapBakeBackend lightmap_bake_backend = LightmapBakeBackend::Gpu;
    bool lightmap_cache_enabled = true;
    bool lightmap_auto_update = true;
    bool lightmap_force_rebake = false;
    char lightmap_cache_path[1024] = {};
    char lightmap_cache_key[1024] = {};
    LightmapBakeProgress* lightmap_bake_progress = nullptr;
    DenoiserMode denoiser_mode = DenoiserMode::Off;
    bool cuda_wavefront = false;
    bool cuda_restir_di = false;
    RestirBiasCorrection cuda_restir_bias_correction = RestirBiasCorrection::Basic;
    bool cuda_restir_final_visibility_reuse = false;
    bool cuda_restir_gi = false;
    RestirBiasCorrection cuda_restir_gi_bias_correction = RestirBiasCorrection::Basic;
    RestirGiResamplingMode cuda_restir_gi_resampling = RestirGiResamplingMode::TemporalSpatial;
    bool cuda_restir_gi_final_mis = true;
    bool cuda_restir_gi_boiling_filter = true;
    float cuda_restir_gi_secondary_roughness = 0.5f;
    bool cuda_restir_pt = false;
    RestirPtResamplingMode cuda_restir_pt_resampling = RestirPtResamplingMode::TemporalSpatial;
    int cuda_restir_pt_max_bounces = 3;
    RestirPtReconnectionMode cuda_restir_pt_reconnection = RestirPtReconnectionMode::Footprint;
    int svgf_iterations = 5;
    float svgf_alpha = 0.05f;
    float svgf_moments_alpha = 0.20f;
    float svgf_phi_color = 4.0f;
    float svgf_phi_normal = 128.0f;
    float svgf_phi_depth = 1.0f;
    bool svgf_rasterized_gbuffer = true;
    DenoiserDebugView svgf_debug_view = DenoiserDebugView::Final;
    AntialiasingMode antialiasing_mode = AntialiasingMode::StablePostAA;
    float camera_jitter_x = 0.0f;
    float camera_jitter_y = 0.0f;
    uint32_t frame_index = 0;
    RenderDirty dirty = RenderDirty::All;
};

inline float halton(uint32_t index, uint32_t base) {
    float result = 0.0f;
    float weight = 1.0f / static_cast<float>(base);
    while (index > 0u) {
        result += weight * static_cast<float>(index % base);
        index /= base;
        weight /= static_cast<float>(base);
    }
    return result;
}

inline Vec2 temporal_jitter(uint32_t frame_index) {
    constexpr float kJitterScale = 0.25f;
    const uint32_t sample = frame_index + 1u;
    return {(halton(sample, 2u) - 0.5f) * kJitterScale, (halton(sample, 3u) - 0.5f) * kJitterScale};
}

inline bool antialiasing_final_view(const RenderSettings& settings) {
    return settings.denoiser_mode == DenoiserMode::Svgf &&
        settings.svgf_debug_view == DenoiserDebugView::Final;
}

inline bool temporal_antialiasing_enabled(const RenderSettings& settings) {
    return antialiasing_final_view(settings) &&
        settings.antialiasing_mode == AntialiasingMode::TAA;
}

inline bool post_antialiasing_enabled(const RenderSettings& settings) {
    return antialiasing_final_view(settings) &&
        settings.antialiasing_mode == AntialiasingMode::StablePostAA;
}

inline bool temporal_jitter_enabled(const RenderSettings& settings) {
    return temporal_antialiasing_enabled(settings) &&
        settings.frame_index > 0u &&
        settings.dirty == RenderDirty::None;
}

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

inline bool irradiance_volume_rendering_enabled(const RenderSettings& settings) {
    return settings.use_irradiance_volume;
}

inline bool lightmap_rendering_enabled(const RenderSettings& settings) {
    return settings.use_lightmap;
}

inline bool svgf_denoising_enabled(const RenderSettings& settings) {
    return settings.denoiser_mode == DenoiserMode::Svgf;
}

inline bool uses_next_event_estimation(const RenderSettings& settings) {
    return settings.sampling_mode == PathSamplingMode::NextEventEstimation ||
        settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling;
}

inline bool uses_multiple_importance_sampling(const RenderSettings& settings) {
    return settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling;
}

struct Framebuffer {
    int width = 0;
    int height = 0;
    std::vector<Vec3> accumulation;
    std::vector<uint32_t> rgba;

    struct AovBuffers {
        std::vector<Vec3> radiance;
        std::vector<Vec3> albedo;
        std::vector<Vec3> emission;
        std::vector<Vec3> normal;
        std::vector<Vec3> world_position;
        std::vector<float> linear_depth;
        std::vector<uint32_t> object_id;
        bool rasterized = false;

        void resize(size_t count) {
            radiance.assign(count, Vec3{});
            albedo.assign(count, Vec3{});
            emission.assign(count, Vec3{});
            normal.assign(count, Vec3{});
            world_position.assign(count, Vec3{});
            linear_depth.assign(count, kInfinity);
            object_id.assign(count, 0u);
            rasterized = false;
        }

        void clear() {
            std::fill(radiance.begin(), radiance.end(), Vec3{});
            std::fill(albedo.begin(), albedo.end(), Vec3{});
            std::fill(emission.begin(), emission.end(), Vec3{});
            std::fill(normal.begin(), normal.end(), Vec3{});
            std::fill(world_position.begin(), world_position.end(), Vec3{});
            std::fill(linear_depth.begin(), linear_depth.end(), kInfinity);
            std::fill(object_id.begin(), object_id.end(), 0u);
            rasterized = false;
        }
    };

    struct SvgfHistory {
        std::vector<Vec3> illumination;
        std::vector<float> moment1;
        std::vector<float> moment2;
        std::vector<float> history_length;
        std::vector<Vec3> normal;
        std::vector<Vec3> world_position;
        std::vector<float> linear_depth;
        std::vector<uint32_t> object_id;
        Camera camera;
        float jitter_x = 0.0f;
        float jitter_y = 0.0f;
        bool valid = false;

        void resize(size_t count) {
            illumination.assign(count, Vec3{});
            moment1.assign(count, 0.0f);
            moment2.assign(count, 0.0f);
            history_length.assign(count, 0.0f);
            normal.assign(count, Vec3{});
            world_position.assign(count, Vec3{});
            linear_depth.assign(count, kInfinity);
            object_id.assign(count, 0u);
            jitter_x = 0.0f;
            jitter_y = 0.0f;
            valid = false;
        }

        void clear() {
            std::fill(illumination.begin(), illumination.end(), Vec3{});
            std::fill(moment1.begin(), moment1.end(), 0.0f);
            std::fill(moment2.begin(), moment2.end(), 0.0f);
            std::fill(history_length.begin(), history_length.end(), 0.0f);
            std::fill(normal.begin(), normal.end(), Vec3{});
            std::fill(world_position.begin(), world_position.end(), Vec3{});
            std::fill(linear_depth.begin(), linear_depth.end(), kInfinity);
            std::fill(object_id.begin(), object_id.end(), 0u);
            jitter_x = 0.0f;
            jitter_y = 0.0f;
            valid = false;
        }
    };

    struct TaaHistory {
        std::vector<Vec3> color;
        std::vector<float> history_length;
        std::vector<Vec3> normal;
        std::vector<Vec3> world_position;
        std::vector<float> linear_depth;
        std::vector<uint32_t> object_id;
        Camera camera;
        float jitter_x = 0.0f;
        float jitter_y = 0.0f;
        bool valid = false;

        void resize(size_t count) {
            color.assign(count, Vec3{});
            history_length.assign(count, 0.0f);
            normal.assign(count, Vec3{});
            world_position.assign(count, Vec3{});
            linear_depth.assign(count, kInfinity);
            object_id.assign(count, 0u);
            jitter_x = 0.0f;
            jitter_y = 0.0f;
            valid = false;
        }

        void clear() {
            std::fill(color.begin(), color.end(), Vec3{});
            std::fill(history_length.begin(), history_length.end(), 0.0f);
            std::fill(normal.begin(), normal.end(), Vec3{});
            std::fill(world_position.begin(), world_position.end(), Vec3{});
            std::fill(linear_depth.begin(), linear_depth.end(), kInfinity);
            std::fill(object_id.begin(), object_id.end(), 0u);
            jitter_x = 0.0f;
            jitter_y = 0.0f;
            valid = false;
        }
    };

    AovBuffers aov;
    SvgfHistory svgf_history;
    TaaHistory taa_history;

    void resize(int w, int h) {
        if (w == width && h == height) {
            return;
        }
        width = w;
        height = h;
        const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
        accumulation.assign(count, Vec3{});
        rgba.assign(count, 0xff000000u);
        aov.resize(count);
        svgf_history.resize(count);
        taa_history.resize(count);
    }

    void clear() {
        std::fill(accumulation.begin(), accumulation.end(), Vec3{});
        std::fill(rgba.begin(), rgba.end(), 0xff000000u);
        aov.clear();
        svgf_history.clear();
        taa_history.clear();
    }

    void clear_accumulation() {
        std::fill(accumulation.begin(), accumulation.end(), Vec3{});
    }
};

struct RasterizedGBufferInterop {
    void* albedo_texture = nullptr;
    void* emission_texture = nullptr;
    void* normal_texture = nullptr;
    void* position_depth_texture = nullptr;
    void* object_id_texture = nullptr;
    int width = 0;
    int height = 0;
    bool valid = false;
};

inline bool has_rasterized_svgf_aov(const Framebuffer& framebuffer) {
    const size_t count = static_cast<size_t>(std::max(0, framebuffer.width)) *
        static_cast<size_t>(std::max(0, framebuffer.height));
    return framebuffer.aov.rasterized &&
        framebuffer.aov.albedo.size() == count &&
        framebuffer.aov.emission.size() == count &&
        framebuffer.aov.normal.size() == count &&
        framebuffer.aov.world_position.size() == count &&
        framebuffer.aov.linear_depth.size() == count &&
        framebuffer.aov.object_id.size() == count;
}

std::shared_ptr<void> build_irradiance_volume_gpu(
    const RenderScene& render_scene, const Scene& scene, const RenderSettings& settings);

std::shared_ptr<void> build_lightmap_gpu(
    const RenderScene& render_scene, const Scene& scene, const RenderSettings& settings);

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual const char* name() const = 0;
    virtual bool available() const = 0;
    virtual void reset() = 0;
    virtual bool has_cached_data() const { return false; }
    virtual void render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) = 0;
};

class CpuPathTracer final : public IRenderer {
public:
    const char* name() const override { return "CPU Path Tracer"; }
    bool available() const override { return true; }
    void reset() override;
    bool has_cached_data() const override { return scene_uploaded_; }
    void render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) override;

private:
    RenderScene cached_render_scene_;
    std::shared_ptr<void> cached_irradiance_volume_;
    std::shared_ptr<void> cached_lightmap_;
    bool scene_uploaded_ = false;
};

class CudaPathTracer final : public IRenderer {
public:
    ~CudaPathTracer() override;
    const char* name() const override { return "CUDA Path Tracer"; }
    bool available() const override;
    void reset() override;
    bool has_cached_data() const override { return scene_uploaded_; }
    void render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) override;
    void render_with_rasterized_gbuffer_interop(
        const Scene& scene,
        const RenderSettings& settings,
        Framebuffer& framebuffer,
        const RasterizedGBufferInterop& interop);
private:
    void release_svgf_gbuffer_interop_cache();
    bool upload_rasterized_gbuffer_interop_to_cuda(
        const RasterizedGBufferInterop& interop,
        int width,
        int height,
        Vec3* albedo,
        Vec3* emission,
        Vec3* normal,
        Vec3* world_position,
        float* depth,
        uint32_t* object_id,
        int& cuda_error_code);
    void render_cpu_fallback(
        const Scene& scene,
        const RenderSettings& settings,
        Framebuffer& framebuffer,
        const char* reason,
        const char* detail = nullptr);

    void* device_accumulation_ = nullptr;
    void* device_rgba_ = nullptr;
    void* device_svgf_radiance_ = nullptr;
    void* device_svgf_albedo_ = nullptr;
    void* device_svgf_emission_ = nullptr;
    void* device_svgf_normal_ = nullptr;
    void* device_svgf_world_position_ = nullptr;
    void* device_svgf_depth_ = nullptr;
    void* device_svgf_object_id_ = nullptr;
    void* device_svgf_history_illumination_ = nullptr;
    void* device_svgf_history_moment1_ = nullptr;
    void* device_svgf_history_moment2_ = nullptr;
    void* device_svgf_history_length_ = nullptr;
    void* device_svgf_history_normal_ = nullptr;
    void* device_svgf_history_world_position_ = nullptr;
    void* device_svgf_history_depth_ = nullptr;
    void* device_svgf_history_object_id_ = nullptr;
    void* device_svgf_temporal_illumination_ = nullptr;
    void* device_svgf_temporal_moment1_ = nullptr;
    void* device_svgf_temporal_moment2_ = nullptr;
    void* device_svgf_temporal_variance_ = nullptr;
    void* device_svgf_temporal_history_length_ = nullptr;
    void* device_svgf_ping_illumination_ = nullptr;
    void* device_svgf_pong_illumination_ = nullptr;
    void* device_svgf_ping_variance_ = nullptr;
    void* device_svgf_pong_variance_ = nullptr;
    void* device_svgf_final_color_ = nullptr;
    void* device_taa_history_color_ = nullptr;
    void* device_taa_history_normal_ = nullptr;
    void* device_taa_history_world_position_ = nullptr;
    void* device_taa_history_depth_ = nullptr;
    void* device_taa_history_object_id_ = nullptr;
    void* device_taa_history_length_ = nullptr;
    void* device_svgf_gbuffer_temp_albedo_ = nullptr;
    void* device_svgf_gbuffer_temp_emission_ = nullptr;
    void* device_svgf_gbuffer_temp_normal_ = nullptr;
    void* device_svgf_gbuffer_temp_position_depth_ = nullptr;
    void* device_scene_ = nullptr;
    void* device_materials_ = nullptr;
    void* device_textures_ = nullptr;
    void* device_triangles_ = nullptr;
    void* device_traversal_triangles_ = nullptr;
    void* device_spheres_ = nullptr;
    void* device_triangle_indices_ = nullptr;
    void* device_light_indices_ = nullptr;
    void* device_directional_lights_ = nullptr;
    void* device_point_lights_ = nullptr;
    void* device_bvh_nodes_ = nullptr;
    void* device_traversal_bvh_nodes_ = nullptr;
    void* device_traversal_bvh8_nodes_ = nullptr;
    void* device_traversal_cwbvh_nodes_ = nullptr;
    void* device_cwbvh_triangle_indices_ = nullptr;
    void* device_cwbvh_triangles_ = nullptr;
    void* device_mesh_instances_ = nullptr;
    void* device_mesh_instance_indices_ = nullptr;
    void* device_tlas_nodes_ = nullptr;
    void* device_traversal_tlas_nodes_ = nullptr;
    void* device_irradiance_volume_directions_ = nullptr;
    void* device_irradiance_volume_irradiance_ = nullptr;
    void* device_irradiance_volume_grids_ = nullptr;
    void* device_irradiance_volume_cells_ = nullptr;
    void* device_irradiance_volume_debug_probes_ = nullptr;
    void* device_lightmap_texels_ = nullptr;
    void* device_wavefront_rays_ = nullptr;
    void* device_wavefront_throughputs_ = nullptr;
    void* device_wavefront_radiance_ = nullptr;
    void* device_wavefront_previous_positions_ = nullptr;
    void* device_wavefront_previous_bsdf_pdfs_ = nullptr;
    void* device_wavefront_rngs_ = nullptr;
    void* device_wavefront_pixels_ = nullptr;
    void* device_wavefront_states_ = nullptr;
    void* device_wavefront_compact_hits_ = nullptr;
    void* device_wavefront_hits_ = nullptr;
    void* device_wavefront_active_indices_ = nullptr;
    void* device_wavefront_next_indices_ = nullptr;
    void* device_wavefront_shade_indices_ = nullptr;
    void* device_wavefront_shadow_indices_ = nullptr;
    void* device_wavefront_gi_indices_ = nullptr;
    void* device_wavefront_bsdf_indices_ = nullptr;
    void* device_wavefront_queue_counters_ = nullptr;
    void* device_wavefront_samples_ = nullptr;
    void* device_wavefront_sample_sum_ = nullptr;
    void* device_restir_initial_ = nullptr;
    void* device_restir_temporal_ = nullptr;
    void* device_restir_history_ = nullptr;
    void* device_restir_scratch_ = nullptr;
    void* device_restir_history_surfaces_ = nullptr;
    void* device_restir_current_surfaces_ = nullptr;
    void* device_restir_temporal_states_ = nullptr;
    void* device_restir_spatial_states_ = nullptr;
    void* device_restir_visibility_rays_ = nullptr;
    void* device_restir_visibility_results_ = nullptr;
    void* device_restir_visibility_indices_ = nullptr;
    void* device_restir_visibility_count_ = nullptr;
    void* device_restir_neighbor_offsets_ = nullptr;
    void* device_restir_environment_pmf_ = nullptr;
    void* device_restir_environment_alias_probability_ = nullptr;
    void* device_restir_environment_alias_index_ = nullptr;
    void* device_restir_light_pmf_ = nullptr;
    void* device_restir_light_alias_probability_ = nullptr;
    void* device_restir_light_alias_index_ = nullptr;
    void* device_restir_gi_initial_samples_ = nullptr;
    void* device_restir_gi_secondary_compact_hits_ = nullptr;
    void* device_restir_gi_secondary_hits_ = nullptr;
    void* device_restir_gi_secondary_results_ = nullptr;
    void* device_restir_gi_secondary_direct_ = nullptr;
    void* device_restir_gi_initial_ = nullptr;
    void* device_restir_gi_temporal_ = nullptr;
    void* device_restir_gi_history_ = nullptr;
    void* device_restir_gi_scratch_ = nullptr;
    void* device_restir_gi_current_surfaces_ = nullptr;
    void* device_restir_gi_history_surfaces_ = nullptr;
    void* device_restir_gi_temporal_states_ = nullptr;
    void* device_restir_gi_spatial_states_ = nullptr;
    void* device_restir_gi_visibility_rays_ = nullptr;
    void* device_restir_gi_visibility_results_ = nullptr;
    void* device_restir_pt_path_states_ = nullptr;
    void* device_restir_pt_compact_hits_ = nullptr;
    void* device_restir_pt_hits_ = nullptr;
    void* device_restir_pt_trace_results_ = nullptr;
    void* device_restir_pt_active_indices_ = nullptr;
    void* device_restir_pt_next_indices_ = nullptr;
    void* device_restir_pt_queue_counters_ = nullptr;
    void* device_restir_pt_nee_reservoirs_ = nullptr;
    void* device_restir_pt_initial_ = nullptr;
    void* device_restir_pt_temporal_ = nullptr;
    void* device_restir_pt_history_ = nullptr;
    void* device_restir_pt_scratch_ = nullptr;
    void* device_restir_pt_current_surfaces_ = nullptr;
    void* device_restir_pt_history_surfaces_ = nullptr;
    void* device_restir_pt_temporal_states_ = nullptr;
    void* device_restir_pt_spatial_states_ = nullptr;
    void* device_restir_pt_visibility_rays_ = nullptr;
    void* device_restir_pt_visibility_results_ = nullptr;
    void* device_restir_pt_sample_ids_ = nullptr;
    void* device_restir_pt_duplication_counts_ = nullptr;
    std::vector<void*> texture_arrays_;
    std::vector<uint64_t> texture_objects_;
    RenderScene cached_render_scene_;
    std::shared_ptr<void> cached_irradiance_volume_;
    std::shared_ptr<void> cached_lightmap_;
    size_t cached_pixels_ = 0;
    int cached_materials_ = 0;
    int cached_textures_ = 0;
    int cached_triangles_ = 0;
    int cached_traversal_triangles_ = 0;
    int cached_spheres_ = 0;
    int cached_triangle_indices_ = 0;
    int cached_lights_ = 0;
    int cached_directional_lights_ = 0;
    int cached_point_lights_ = 0;
    int cached_bvh_nodes_ = 0;
    int cached_traversal_bvh_nodes_ = 0;
    int cached_traversal_bvh8_nodes_ = 0;
    int cached_traversal_cwbvh_nodes_ = 0;
    int cached_cwbvh_triangle_indices_ = 0;
    int cached_cwbvh_triangles_ = 0;
    int cached_mesh_instances_ = 0;
    int cached_mesh_instance_indices_ = 0;
    int cached_tlas_nodes_ = 0;
    int cached_traversal_tlas_nodes_ = 0;
    int cached_irradiance_volume_directions_ = 0;
    int cached_irradiance_volume_irradiance_ = 0;
    int cached_irradiance_volume_grids_ = 0;
    int cached_irradiance_volume_cells_ = 0;
    int cached_irradiance_volume_debug_probes_ = 0;
    int cached_lightmap_texels_ = 0;
    bool scene_uploaded_ = false;
    bool cached_render_scene_valid_ = false;
    bool cached_irradiance_volume_enabled_ = false;
    bool cached_lightmap_enabled_ = false;
    bool svgf_history_valid_ = false;
    Camera svgf_history_camera_;
    float svgf_history_jitter_x_ = 0.0f;
    float svgf_history_jitter_y_ = 0.0f;
    bool taa_history_valid_ = false;
    Camera taa_history_camera_;
    float taa_history_jitter_x_ = 0.0f;
    float taa_history_jitter_y_ = 0.0f;
    bool restir_history_valid_ = false;
    bool restir_was_enabled_ = false;
    Camera restir_history_camera_;
    int cached_restir_environment_texels_ = 0;
    int cached_restir_light_count_ = 0;
    RestirBiasCorrection restir_history_bias_correction_ = RestirBiasCorrection::Basic;
    bool restir_history_visibility_reuse_ = false;
    bool restir_gi_history_valid_ = false;
    bool restir_gi_was_enabled_ = false;
    Camera restir_gi_history_camera_;
    RestirBiasCorrection restir_gi_history_bias_correction_ = RestirBiasCorrection::Basic;
    RestirGiResamplingMode restir_gi_history_resampling_ = RestirGiResamplingMode::TemporalSpatial;
    bool restir_gi_history_final_mis_ = true;
    bool restir_gi_history_boiling_filter_ = true;
    float restir_gi_history_secondary_roughness_ = 0.5f;
    bool restir_pt_history_valid_ = false;
    bool restir_pt_was_enabled_ = false;
    Camera restir_pt_history_camera_;
    RestirPtResamplingMode restir_pt_history_resampling_ = RestirPtResamplingMode::TemporalSpatial;
    RestirPtReconnectionMode restir_pt_history_reconnection_ = RestirPtReconnectionMode::Footprint;
    int restir_pt_history_max_bounces_ = 3;
    RasterizedGBufferInterop svgf_gbuffer_interop_;
    void* svgf_gbuffer_cuda_resources_[5] = {};
    void* svgf_gbuffer_d3d_resources_[5] = {};
    int svgf_gbuffer_width_ = 0;
    int svgf_gbuffer_height_ = 0;
    std::vector<std::string> reported_fallback_reasons_;
};

} // namespace lt

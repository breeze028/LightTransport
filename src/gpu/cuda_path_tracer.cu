#include "lt/renderer.h"
#include "lt/log.h"

#if LT_HAS_CUDA

#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define LT_HAS_NVTX3 1
#else
#define LT_HAS_NVTX3 0
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <xatlas.h>

#define TINYBVH_IMPLEMENTATION
#include <tiny_bvh.h>

namespace lt {
namespace {

class ScopedNvtxRange {
public:
    explicit ScopedNvtxRange(const char* name) {
#if LT_HAS_NVTX3
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~ScopedNvtxRange() {
#if LT_HAS_NVTX3
        nvtxRangePop();
#endif
    }
    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
};

#include "types.cuh"
#include "math.cuh"
#include "intersection.cuh"
#include "shading.cuh"
#include "kernel.cuh"
#include "scene_upload.cuh"
#include "../cpu/types.inl"
#include "../cpu/intersection.inl"
#include "../cpu/irradiance_volume.inl"
#include "../cpu/lightmap.inl"
#include "../cpu/shading.inl"
#include "../cpu/irradiance_volume_bake.inl"
#include "../cpu/lightmap_bake.inl"

template <bool TransparentVisibility>
void launch_restir_visibility_trace(const GpuScene* scene, GpuWavefrontPaths paths,
    const int* visibility_indices, const int* visibility_count,
    const GpuRestirVisibilityRay* visibility_rays, int* visibility_results,
    bool two_level, bool wide_bvh, bool cwbvh, int grid_size, int block_size) {
    #define LT_LAUNCH_RESTIR_VISIBILITY_TRACE(TWO_LEVEL, LAYOUT) \
        restir_trace_visibility_kernel<TransparentVisibility, TWO_LEVEL, LAYOUT><<<grid_size, block_size>>>( \
            scene, paths, visibility_indices, visibility_count, visibility_rays, visibility_results)
    if (two_level && cwbvh) {
        LT_LAUNCH_RESTIR_VISIBILITY_TRACE(true, GpuTraversalLayout::CwBvh);
    } else if (two_level && wide_bvh) {
        LT_LAUNCH_RESTIR_VISIBILITY_TRACE(true, GpuTraversalLayout::Bvh8);
    } else if (two_level) {
        LT_LAUNCH_RESTIR_VISIBILITY_TRACE(true, GpuTraversalLayout::Binary);
    } else if (wide_bvh) {
        LT_LAUNCH_RESTIR_VISIBILITY_TRACE(false, GpuTraversalLayout::Bvh8);
    } else {
        LT_LAUNCH_RESTIR_VISIBILITY_TRACE(false, GpuTraversalLayout::Binary);
    }
    #undef LT_LAUNCH_RESTIR_VISIBILITY_TRACE
}

template <bool AlphaVisibility>
void launch_restir_gi_secondary_trace(const GpuScene* scene, RenderSettings settings,
    GpuWavefrontPaths paths, const int* bsdf_indices, const GpuWavefrontQueueCounters* queue_counters,
    const GpuRestirGiInitialSample* initial_samples, GpuCompactHit* secondary_hits, int* results,
    bool two_level, bool wide_bvh, bool cwbvh, int grid_size, int block_size) {
    #define LT_LAUNCH_RESTIR_GI_SECONDARY_TRACE(TWO_LEVEL, LAYOUT) \
        restir_gi_trace_secondary_kernel<AlphaVisibility, TWO_LEVEL, LAYOUT><<<grid_size, block_size>>>( \
            scene, settings, paths, bsdf_indices, queue_counters, initial_samples, secondary_hits, results)
    if (two_level && cwbvh) {
        LT_LAUNCH_RESTIR_GI_SECONDARY_TRACE(true, GpuTraversalLayout::CwBvh);
    } else if (two_level && wide_bvh) {
        LT_LAUNCH_RESTIR_GI_SECONDARY_TRACE(true, GpuTraversalLayout::Bvh8);
    } else if (two_level) {
        LT_LAUNCH_RESTIR_GI_SECONDARY_TRACE(true, GpuTraversalLayout::Binary);
    } else if (wide_bvh) {
        LT_LAUNCH_RESTIR_GI_SECONDARY_TRACE(false, GpuTraversalLayout::Bvh8);
    } else {
        LT_LAUNCH_RESTIR_GI_SECONDARY_TRACE(false, GpuTraversalLayout::Binary);
    }
    #undef LT_LAUNCH_RESTIR_GI_SECONDARY_TRACE
}

template <bool AlphaVisibility>
void launch_restir_pt_trace(const GpuScene* scene, GpuRestirPtPathState* states,
    GpuCompactHit* compact_hits, int* results, const int* active_indices,
    const GpuRestirPtQueueCounters* counters, bool two_level, bool wide_bvh,
    bool cwbvh, int grid_size, int block_size) {
    #define LT_LAUNCH_RESTIR_PT_TRACE(TWO_LEVEL, LAYOUT) \
        restir_pt_trace_kernel<AlphaVisibility, TWO_LEVEL, LAYOUT><<<grid_size, block_size>>>( \
            scene, states, compact_hits, results, active_indices, counters)
    if (two_level && cwbvh) {
        LT_LAUNCH_RESTIR_PT_TRACE(true, GpuTraversalLayout::CwBvh);
    } else if (two_level && wide_bvh) {
        LT_LAUNCH_RESTIR_PT_TRACE(true, GpuTraversalLayout::Bvh8);
    } else if (two_level) {
        LT_LAUNCH_RESTIR_PT_TRACE(true, GpuTraversalLayout::Binary);
    } else if (wide_bvh) {
        LT_LAUNCH_RESTIR_PT_TRACE(false, GpuTraversalLayout::Bvh8);
    } else {
        LT_LAUNCH_RESTIR_PT_TRACE(false, GpuTraversalLayout::Binary);
    }
    #undef LT_LAUNCH_RESTIR_PT_TRACE
}

struct GBufferFloat4 {
    float x, y, z, w;
};

__global__ void unpack_rasterized_gbuffer_kernel(
    const GBufferFloat4* albedo4,
    const GBufferFloat4* emission4,
    const GBufferFloat4* normal4,
    const GBufferFloat4* position_depth4,
    Vec3* albedo,
    Vec3* emission,
    Vec3* normal,
    Vec3* world_position,
    float* depth,
    int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const GBufferFloat4 a = albedo4[idx];
    const GBufferFloat4 e = emission4[idx];
    const GBufferFloat4 n = normal4[idx];
    const GBufferFloat4 p = position_depth4[idx];
    albedo[idx] = {a.x, a.y, a.z};
    emission[idx] = {e.x, e.y, e.z};
    normal[idx] = {n.x, n.y, n.z};
    world_position[idx] = {p.x, p.y, p.z};
    depth[idx] = p.w;
}

struct PackedGpuIrradianceVolume {
    GpuIrradianceVolume volume;
    std::vector<Vec3> directions;
    std::vector<Vec3> irradiance;
    std::vector<GpuIrradianceVolumeGrid> grids;
    std::vector<int> cell_subgrid_indices;
    std::vector<GpuIrradianceVolumeDebugProbe> debug_probes;
};

struct PackedGpuLightmap {
    GpuLightmap lightmap;
    std::vector<Vec3> texels;
};

struct PackedGpuEnvironmentSampler {
    GpuEnvironmentSampler sampler;
    std::vector<float> pmf;
    std::vector<float> alias_probability;
    std::vector<int> alias_index;
};

struct PackedGpuLightSampler {
    GpuLightSampler sampler;
    std::vector<float> pmf;
    std::vector<float> alias_probability;
    std::vector<int> alias_index;
};

std::vector<Vec2> build_restir_neighbor_offsets() {
    std::vector<Vec2> offsets(static_cast<size_t>(kRestirNeighborOffsetCount));
    uint32_t state = 0x6d2b79f5u;
    const auto next_random = [&state]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (static_cast<float>(state) + 0.5f) * (1.0f / 4294967296.0f);
    };
    for (Vec2& offset : offsets) {
        const float radius = std::sqrt(next_random());
        const float angle = 2.0f * kPi * next_random();
        offset = {radius * std::cos(angle), radius * std::sin(angle)};
    }
    return offsets;
}

float environment_sampler_luminance(Vec3 color) {
    return std::max(0.0f, color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f);
}

bool build_alias_distribution(const std::vector<float>& weights, std::vector<float>& pmf,
    std::vector<float>& alias_probability, std::vector<int>& alias_index) {
    pmf.clear();
    alias_probability.clear();
    alias_index.clear();
    if (weights.empty()) {
        return true;
    }
    float total = 0.0f;
    for (float weight : weights) {
        total += std::max(0.0f, weight);
    }
    if (!(total > 0.0f) || !std::isfinite(total)) {
        return false;
    }
    const int count = static_cast<int>(weights.size());
    pmf.resize(weights.size());
    alias_probability.assign(weights.size(), 1.0f);
    alias_index.resize(weights.size());
    std::vector<float> scaled(weights.size());
    std::vector<int> underfull;
    std::vector<int> large;
    underfull.reserve(weights.size());
    large.reserve(weights.size());
    for (int i = 0; i < count; ++i) {
        pmf[static_cast<size_t>(i)] = std::max(0.0f, weights[static_cast<size_t>(i)]) / total;
        scaled[static_cast<size_t>(i)] = pmf[static_cast<size_t>(i)] * static_cast<float>(count);
        alias_index[static_cast<size_t>(i)] = i;
        (scaled[static_cast<size_t>(i)] < 1.0f ? underfull : large).push_back(i);
    }
    while (!underfull.empty() && !large.empty()) {
        const int lesser = underfull.back();
        underfull.pop_back();
        const int greater = large.back();
        large.pop_back();
        alias_probability[static_cast<size_t>(lesser)] = scaled[static_cast<size_t>(lesser)];
        alias_index[static_cast<size_t>(lesser)] = greater;
        scaled[static_cast<size_t>(greater)] = (scaled[static_cast<size_t>(greater)] +
            scaled[static_cast<size_t>(lesser)]) - 1.0f;
        (scaled[static_cast<size_t>(greater)] < 1.0f ? underfull : large).push_back(greater);
    }
    return true;
}

bool build_restir_light_sampler(const Scene& scene, const RenderScene& render_scene,
    PackedGpuLightSampler& packed) {
    packed = {};
    std::vector<float> weights;
    weights.reserve(render_scene.light_triangle_indices.size());
    for (int triangle_index : render_scene.light_triangle_indices) {
        if (triangle_index < 0 || triangle_index >= static_cast<int>(render_scene.triangles.size())) {
            return false;
        }
        const Triangle& triangle = render_scene.triangles[static_cast<size_t>(triangle_index)];
        if (triangle.material < 0 || triangle.material >= static_cast<int>(scene.materials.size())) {
            return false;
        }
        Vec3 emission = scene.materials[static_cast<size_t>(triangle.material)]->emitted(
            (triangle.uv0 + triangle.uv1 + triangle.uv2) * (1.0f / 3.0f));
        bool double_sided = scene.materials[static_cast<size_t>(triangle.material)]->double_sided;
        if (triangle.mesh >= 0 && triangle.mesh < static_cast<int>(scene.meshes.size())) {
            const LightComponent& light = scene.meshes[static_cast<size_t>(triangle.mesh)].light;
            if (light.enabled && light.intensity > 0.0f) {
                emission = light.color * light.intensity;
                double_sided = light.double_sided;
            }
        }
        const float area = 0.5f * length(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        const float sidedness = double_sided ? 2.0f : 1.0f;
        weights.push_back(std::max(1.0e-8f, environment_sampler_luminance(emission) * area * sidedness));
    }
    if (!build_alias_distribution(weights, packed.pmf, packed.alias_probability, packed.alias_index)) {
        return false;
    }
    packed.sampler.light_count = static_cast<int>(weights.size());
    return true;
}

bool build_environment_sampler(const Scene& scene, PackedGpuEnvironmentSampler& packed) {
    packed = {};
    const std::shared_ptr<Texture>& texture = scene.environment.texture;
    if (!texture || texture->width <= 0 || texture->height <= 0 ||
        texture->pixels.size() != static_cast<size_t>(texture->width) * static_cast<size_t>(texture->height)) {
        return true;
    }
    const int width = texture->width;
    const int height = texture->height;
    const int count = width * height;
    packed.pmf.resize(static_cast<size_t>(count));
    std::vector<float> scaled(static_cast<size_t>(count));
    float total = 0.0f;
    for (int y = 0; y < height; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        const float solid_angle_weight = scene.environment.mapping == Environment::Mapping::Equirectangular
            ? std::sin((1.0f - v) * kPi) : 1.0f;
        for (int x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const Vec3 radiance = texture->pixels[index] * scene.environment.color * scene.environment.strength;
            const float weight = std::max(1.0e-8f, environment_sampler_luminance(radiance) * solid_angle_weight);
            packed.pmf[index] = weight;
            total += weight;
        }
    }
    if (!(total > 0.0f) || !std::isfinite(total)) {
        return false;
    }
    packed.alias_probability.assign(static_cast<size_t>(count), 1.0f);
    packed.alias_index.resize(static_cast<size_t>(count));
    std::vector<int> underfull;
    std::vector<int> large;
    underfull.reserve(static_cast<size_t>(count));
    large.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        packed.pmf[static_cast<size_t>(i)] /= total;
        scaled[static_cast<size_t>(i)] = packed.pmf[static_cast<size_t>(i)] * static_cast<float>(count);
        packed.alias_index[static_cast<size_t>(i)] = i;
        (scaled[static_cast<size_t>(i)] < 1.0f ? underfull : large).push_back(i);
    }
    while (!underfull.empty() && !large.empty()) {
        const int lesser = underfull.back();
        underfull.pop_back();
        const int greater = large.back();
        large.pop_back();
        packed.alias_probability[static_cast<size_t>(lesser)] = scaled[static_cast<size_t>(lesser)];
        packed.alias_index[static_cast<size_t>(lesser)] = greater;
        scaled[static_cast<size_t>(greater)] = (scaled[static_cast<size_t>(greater)] +
            scaled[static_cast<size_t>(lesser)]) - 1.0f;
        (scaled[static_cast<size_t>(greater)] < 1.0f ? underfull : large).push_back(greater);
    }
    packed.sampler.width = width;
    packed.sampler.height = height;
    packed.sampler.texel_count = count;
    return true;
}

int flatten_irradiance_grid(const IrradianceVolumeGrid& grid, int direction_count, PackedGpuIrradianceVolume& packed) {
    if (!size_fits_int(packed.grids.size()) || !size_fits_int(packed.irradiance.size() / static_cast<size_t>(std::max(1, direction_count))) ||
        !size_fits_int(packed.cell_subgrid_indices.size())) {
        return -1;
    }
    const int grid_index = static_cast<int>(packed.grids.size());
    GpuIrradianceVolumeGrid gpu_grid;
    gpu_grid.bounds_min = grid.bounds.min;
    gpu_grid.bounds_max = grid.bounds.max;
    gpu_grid.resolution = grid.resolution;
    gpu_grid.sample_offset = static_cast<int>(packed.irradiance.size() / static_cast<size_t>(std::max(1, direction_count)));
    gpu_grid.cell_offset = static_cast<int>(packed.cell_subgrid_indices.size());
    gpu_grid.cell_count = static_cast<int>(grid.cells.size());
    packed.grids.push_back(gpu_grid);

    if (direction_count > 0) {
        for (const IrradianceVolumeSample& sample : grid.samples) {
            for (int direction_index = 0; direction_index < direction_count; ++direction_index) {
                const size_t index = static_cast<size_t>(direction_index);
                packed.irradiance.push_back(index < sample.irradiance.size() ? sample.irradiance[index] : Vec3{});
            }
        }
    }

    const int cell_offset = gpu_grid.cell_offset;
    packed.cell_subgrid_indices.resize(packed.cell_subgrid_indices.size() + grid.cells.size(), -1);
    for (int cell_index = 0; cell_index < static_cast<int>(grid.cells.size()); ++cell_index) {
        const IrradianceVolumeCell& cell = grid.cells[static_cast<size_t>(cell_index)];
        if (!cell.subgrid) {
            continue;
        }
        const int subgrid_index = flatten_irradiance_grid(*cell.subgrid, direction_count, packed);
        if (subgrid_index < 0) {
            return -1;
        }
        packed.cell_subgrid_indices[static_cast<size_t>(cell_offset + cell_index)] = subgrid_index;
    }
    return grid_index;
}

bool pack_irradiance_volume(const IrradianceVolume* volume, PackedGpuIrradianceVolume& packed) {
    packed = {};
    if (!volume || volume->directions.empty()) {
        return true;
    }
    if (!size_fits_int(volume->directions.size()) || !size_fits_int(volume->debug_probes.size())) {
        return false;
    }
    packed.directions = volume->directions;
    packed.debug_probes.reserve(volume->debug_probes.size());
    for (const IrradianceVolumeDebugProbe& probe : volume->debug_probes) {
        packed.debug_probes.push_back({probe.position, probe.radius});
    }

    const int direction_count = static_cast<int>(packed.directions.size());
    if (flatten_irradiance_grid(volume->grid, direction_count, packed) != 0) {
        return false;
    }
    if (!size_fits_int(packed.grids.size()) || !size_fits_int(packed.cell_subgrid_indices.size()) ||
        !size_fits_int(packed.irradiance.size() / static_cast<size_t>(std::max(1, direction_count)))) {
        return false;
    }

    packed.volume.direction_count = direction_count;
    packed.volume.grid_count = static_cast<int>(packed.grids.size());
    packed.volume.irradiance_sample_count = direction_count > 0
        ? static_cast<int>(packed.irradiance.size() / static_cast<size_t>(direction_count))
        : 0;
    packed.volume.cell_count = static_cast<int>(packed.cell_subgrid_indices.size());
    packed.volume.debug_probe_count = static_cast<int>(packed.debug_probes.size());
    return true;
}

bool pack_lightmap(const Lightmap* lightmap, PackedGpuLightmap& packed) {
    packed = {};
    if (!lightmap || lightmap->width <= 0 || lightmap->height <= 0) {
        return true;
    }
    if (!size_fits_int(lightmap->texels.size())) {
        return false;
    }
    packed.texels = lightmap->texels;
    packed.lightmap.width = lightmap->width;
    packed.lightmap.height = lightmap->height;
    return true;
}

const char* cuda_error_text(cudaError_t error) {
    return error == cudaSuccess ? nullptr : cudaGetErrorString(error);
}

void configure_cuda_wait_policy_once() {
    static bool configured = false;
    if (configured) {
        return;
    }
    configured = true;
    const cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (error != cudaSuccess && error != cudaErrorSetOnActiveProcess) {
        LT_LOG_DEBUG("CUDA blocking wait policy unavailable: {}", cudaGetErrorString(error));
    }
}

#include "irradiance_volume_bake.cuh"
#include "lightmap_bake.cuh"

} // namespace

std::shared_ptr<void> build_irradiance_volume_gpu(
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings)
{
    const auto begin = std::chrono::steady_clock::now();

    // 1. Build volume layout on CPU (directions, weights, sparse grid).
    auto volume = std::make_shared<IrradianceVolume>();
    volume->grid_resolution = std::max(2, settings.irradiance_volume_grid_resolution);
    volume->subgrid_resolution = std::max(2, settings.irradiance_volume_subgrid_resolution);
    volume->direction_resolution = std::max(1, settings.irradiance_volume_direction_resolution);
    volume->bake_samples = std::max(1, settings.irradiance_volume_bake_samples);
    volume->bake_bounces = std::max(1, settings.irradiance_volume_bake_bounces);
    volume->bounds = scene_irradiance_bounds(render_scene, settings);
    volume->directions = make_irradiance_volume_directions(volume->direction_resolution);
    volume->cosine_weights = make_irradiance_volume_weights(volume->directions);

    initialize_irradiance_grid(*volume, volume->grid, render_scene,
        volume->bounds, volume->grid_resolution, true);
    collect_debug_probes_from_grid(*volume, volume->grid);
    volume->unique_debug_probe_count = volume->debug_probes.size();

    // Collect flat probe position list.
    std::vector<Vec3> probe_positions;
    probe_positions.reserve(volume->spatial_sample_count);
    collect_probe_positions(volume->grid, probe_positions);

    const int probe_count = static_cast<int>(probe_positions.size());
    const int direction_count = static_cast<int>(volume->directions.size());
    if (probe_count == 0 || direction_count == 0) {
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        if (settings.irradiance_volume_bake_progress) {
            IrradianceVolumeBakeProgress& p = *settings.irradiance_volume_bake_progress;
            p.total_samples.store(volume->spatial_sample_count, std::memory_order_relaxed);
            p.completed_samples.store(volume->spatial_sample_count, std::memory_order_relaxed);
            p.direction_count.store(direction_count, std::memory_order_relaxed);
            p.elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
            p.phase.store(static_cast<int>(IrradianceVolumeBakePhase::Complete), std::memory_order_relaxed);
        }
        return volume;
    }

    // 2. Pack and upload the bake scene to GPU.
    PackedGpuScene packed;
    if (!pack_scene_from_render_scene(scene, settings, render_scene, packed)) {
        LT_LOG_ERROR("GPU irradiance volume bake: failed to pack scene");
        return nullptr;
    }

    // Temporary device pointers (separate from CudaPathTracer's cache).
    void* dev_scene = nullptr;
    void* dev_materials = nullptr;
    void* dev_triangles = nullptr;
    void* dev_traversal_triangles = nullptr;
    void* dev_spheres = nullptr;
    void* dev_triangle_indices = nullptr;
    void* dev_light_indices = nullptr;
    void* dev_directional_lights = nullptr;
    void* dev_point_lights = nullptr;
    void* dev_bvh_nodes = nullptr;
    void* dev_traversal_bvh_nodes = nullptr;
    void* dev_mesh_instances = nullptr;
    void* dev_mesh_instance_indices = nullptr;
    void* dev_tlas_nodes = nullptr;
    void* dev_traversal_tlas_nodes = nullptr;
    void* dev_textures_buf = nullptr;
    int hint_materials = 0, hint_triangles = 0, hint_traversal_triangles = 0, hint_spheres = 0;
    int hint_triangle_indices = 0, hint_light_indices = 0;
    int hint_directional_lights = 0, hint_point_lights = 0, hint_bvh_nodes = 0, hint_traversal_bvh_nodes = 0;
    int hint_mesh_instances = 0, hint_mesh_instance_indices = 0;
    int hint_tlas_nodes = 0, hint_traversal_tlas_nodes = 0, hint_textures = 0;
    std::vector<void*> texture_arrays;
    std::vector<uint64_t> texture_objects;

    cudaError_t cuda_error = cudaSuccess;
    auto cleanup_scene = [&]() {
        cudaFree(dev_scene);
        cudaFree(dev_materials);
        cudaFree(dev_triangles);
        cudaFree(dev_traversal_triangles);
        cudaFree(dev_spheres);
        cudaFree(dev_triangle_indices);
        cudaFree(dev_light_indices);
        cudaFree(dev_directional_lights);
        cudaFree(dev_point_lights);
        cudaFree(dev_bvh_nodes);
        cudaFree(dev_traversal_bvh_nodes);
        cudaFree(dev_mesh_instances);
        cudaFree(dev_mesh_instance_indices);
        cudaFree(dev_tlas_nodes);
        cudaFree(dev_traversal_tlas_nodes);
        cudaFree(dev_textures_buf);
        release_texture_objects(texture_arrays, texture_objects);
    };

    if ((cuda_error = cudaMalloc(&dev_scene, sizeof(GpuScene))) != cudaSuccess) {
        LT_LOG_ERROR("GPU irradiance volume bake: cudaMalloc scene failed: {}", cudaGetErrorString(cuda_error));
        cleanup_scene();
        return nullptr;
    }

    if (!upload_buffer(dev_materials, hint_materials, packed.materials) ||
        !upload_buffer(dev_triangles, hint_triangles, packed.triangles) ||
        !upload_buffer(dev_traversal_triangles, hint_traversal_triangles, packed.traversal_triangles) ||
        !upload_buffer(dev_spheres, hint_spheres, packed.spheres) ||
        !upload_buffer(dev_triangle_indices, hint_triangle_indices, packed.triangle_indices) ||
        !upload_buffer(dev_light_indices, hint_light_indices, packed.light_indices) ||
        !upload_buffer(dev_directional_lights, hint_directional_lights, packed.directional_lights) ||
        !upload_buffer(dev_point_lights, hint_point_lights, packed.point_lights) ||
        !upload_buffer(dev_bvh_nodes, hint_bvh_nodes, packed.bvh_nodes) ||
        !upload_buffer(dev_traversal_bvh_nodes, hint_traversal_bvh_nodes, packed.traversal_bvh_nodes) ||
        !upload_buffer(dev_mesh_instances, hint_mesh_instances, packed.mesh_instances) ||
        !upload_buffer(dev_mesh_instance_indices, hint_mesh_instance_indices, packed.mesh_instance_indices) ||
        !upload_buffer(dev_tlas_nodes, hint_tlas_nodes, packed.tlas_nodes) ||
        !upload_buffer(dev_traversal_tlas_nodes, hint_traversal_tlas_nodes, packed.traversal_tlas_nodes) ||
        !upload_buffer(dev_textures_buf, hint_textures, packed.textures) ||
        !upload_texture_objects(scene, packed, texture_arrays, texture_objects)) {
        LT_LOG_ERROR("GPU irradiance volume bake: scene upload failed");
        cleanup_scene();
        return nullptr;
    }

    packed.scene.materials = static_cast<GpuMaterial*>(dev_materials);
    packed.scene.textures = static_cast<GpuTexture*>(dev_textures_buf);
    packed.scene.triangles = static_cast<GpuTriangle*>(dev_triangles);
    packed.scene.traversal_triangles = static_cast<GpuTraversalTriangle*>(dev_traversal_triangles);
    packed.scene.spheres = static_cast<GpuSphere*>(dev_spheres);
    packed.scene.triangle_indices = static_cast<int*>(dev_triangle_indices);
    packed.scene.light_indices = static_cast<int*>(dev_light_indices);
    packed.scene.directional_lights = static_cast<GpuDirectionalLight*>(dev_directional_lights);
    packed.scene.point_lights = static_cast<GpuPointLight*>(dev_point_lights);
    packed.scene.bvh_nodes = static_cast<GpuBvhNode*>(dev_bvh_nodes);
    packed.scene.traversal_bvh_nodes = static_cast<GpuTraversalBvhNode*>(dev_traversal_bvh_nodes);
    packed.scene.mesh_instances = static_cast<GpuMeshInstance*>(dev_mesh_instances);
    packed.scene.mesh_instance_indices = static_cast<int*>(dev_mesh_instance_indices);
    packed.scene.tlas_nodes = static_cast<GpuBvhNode*>(dev_tlas_nodes);
    packed.scene.traversal_tlas_nodes = static_cast<GpuTraversalBvhNode*>(dev_traversal_tlas_nodes);

    cuda_error = cudaMemcpy(static_cast<GpuScene*>(dev_scene), &packed.scene,
        sizeof(GpuScene), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU irradiance volume bake: scene header upload failed: {}",
            cudaGetErrorString(cuda_error));
        cleanup_scene();
        return nullptr;
    }

    // 3. Upload probe data.
    void* dev_probe_positions = nullptr;
    void* dev_directions = nullptr;
    void* dev_cosine_weights = nullptr;
    void* dev_probe_radiance = nullptr;
    void* dev_probe_irradiance = nullptr;
    int hint_probes = 0, hint_dirs = 0, hint_weights = 0;

    auto cleanup_all = [&]() {
        cleanup_scene();
        cudaFree(dev_probe_positions);
        cudaFree(dev_directions);
        cudaFree(dev_cosine_weights);
        cudaFree(dev_probe_radiance);
        cudaFree(dev_probe_irradiance);
    };

    if (!upload_buffer(dev_probe_positions, hint_probes, probe_positions) ||
        !upload_buffer(dev_directions, hint_dirs, volume->directions) ||
        !upload_buffer(dev_cosine_weights, hint_weights, volume->cosine_weights)) {
        LT_LOG_ERROR("GPU irradiance volume bake: probe data upload failed");
        cleanup_all();
        return nullptr;
    }

    const size_t radiance_count = static_cast<size_t>(probe_count) * static_cast<size_t>(direction_count);
    if ((cuda_error = cudaMalloc(&dev_probe_radiance, radiance_count * sizeof(Vec3))) != cudaSuccess ||
        (cuda_error = cudaMalloc(&dev_probe_irradiance, radiance_count * sizeof(Vec3))) != cudaSuccess) {
        LT_LOG_ERROR("GPU irradiance volume bake: radiance buffer alloc failed: {}",
            cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }

    // 4. Set progress.
    if (settings.irradiance_volume_bake_progress) {
        IrradianceVolumeBakeProgress& p = *settings.irradiance_volume_bake_progress;
        p.total_samples.store(static_cast<uint64_t>(probe_count), std::memory_order_relaxed);
        p.completed_samples.store(0, std::memory_order_relaxed);
        p.total_rays.store(static_cast<uint64_t>(probe_count) * static_cast<uint64_t>(direction_count) *
            static_cast<uint64_t>(volume->bake_samples), std::memory_order_relaxed);
        p.traced_rays.store(0, std::memory_order_relaxed);
        p.direction_count.store(direction_count, std::memory_order_relaxed);
        p.elapsed_ms.store(0.0, std::memory_order_relaxed);
        p.phase.store(static_cast<int>(IrradianceVolumeBakePhase::Baking), std::memory_order_relaxed);
    }

    // 5. Launch radiance kernel.
    const int total_threads = probe_count * direction_count;
    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;
    bake_radiance_kernel<<<grid_size, block_size>>>(
        static_cast<const GpuScene*>(dev_scene),
        static_cast<const Vec3*>(dev_probe_positions),
        probe_count,
        static_cast<const Vec3*>(dev_directions),
        direction_count,
        volume->bake_samples,
        volume->bake_bounces,
        static_cast<int>(settings.sampling_mode),
        static_cast<int>(settings.mis_heuristic),
        static_cast<int>(settings.acceleration_structure),
        std::max(0.0f, settings.emissive_intensity_scale),
        static_cast<Vec3*>(dev_probe_radiance));

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU irradiance volume bake: radiance kernel failed: {}",
            cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }

    // Update progress after radiance pass.
    if (settings.irradiance_volume_bake_progress) {
        IrradianceVolumeBakeProgress& p = *settings.irradiance_volume_bake_progress;
        p.traced_rays.store(static_cast<uint64_t>(probe_count) *
            static_cast<uint64_t>(direction_count) * static_cast<uint64_t>(volume->bake_samples),
            std::memory_order_relaxed);
        p.completed_samples.store(static_cast<uint64_t>(probe_count), std::memory_order_relaxed);
    }

    // 6. Launch reduce kernel.
    reduce_irradiance_kernel<<<grid_size, block_size>>>(
        static_cast<const Vec3*>(dev_probe_radiance),
        probe_count,
        direction_count,
        static_cast<const float*>(dev_cosine_weights),
        static_cast<Vec3*>(dev_probe_irradiance),
        1024.0f);

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU irradiance volume bake: reduce kernel failed: {}",
            cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }

    // 7. Download results and populate volume.
    std::vector<Vec3> irradiance_results(radiance_count);
    cuda_error = cudaMemcpy(irradiance_results.data(), dev_probe_irradiance,
        radiance_count * sizeof(Vec3), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU irradiance volume bake: result download failed: {}",
            cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }

    int sample_idx = 0;
    populate_irradiance_results(volume->grid, irradiance_results, direction_count, sample_idx);

    // 8. Cleanup temporary device allocations.
    cleanup_all();

    // 9. Final progress and log.
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    if (settings.irradiance_volume_bake_progress) {
        IrradianceVolumeBakeProgress& p = *settings.irradiance_volume_bake_progress;
        p.elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
        p.phase.store(static_cast<int>(IrradianceVolumeBakePhase::Complete), std::memory_order_relaxed);
    }

    const size_t irradiance_bytes = volume->spatial_sample_count *
        static_cast<size_t>(direction_count) * sizeof(Vec3);
    const size_t weight_bytes = volume->cosine_weights.size() * sizeof(float);
    const uint64_t total_rays = static_cast<uint64_t>(probe_count) * static_cast<uint64_t>(direction_count) *
        static_cast<uint64_t>(volume->bake_samples);
    LT_LOG_INFO(
        "Irradiance volume baked on GPU: samples={} debug_probes={} first_cells={} subgrids={} directions={} rays={} memory_kib={} elapsed_ms={} elapsed_s={}",
        volume->spatial_sample_count,
        volume->unique_debug_probe_count,
        volume->first_level_cell_count,
        volume->subgrid_count,
        direction_count,
        total_rays,
        (irradiance_bytes + weight_bytes) / 1024u,
        format_decimal(elapsed_ms, 3),
        format_decimal(elapsed_ms * 0.001, 2));

    return volume;
}

std::shared_ptr<void> build_lightmap_gpu(
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings)
{
    const auto begin = std::chrono::steady_clock::now();

    // 1. UV unwrap on CPU using xatlas.
    auto lightmap = std::make_shared<Lightmap>();
    if (!generate_lightmap_uvs(render_scene, settings, *lightmap)) {
        return nullptr;
    }

    // 2. Raster texels to get world positions and normals.
    std::vector<GpuLightmapBakeTexel> bake_texels = raster_lightmap_texels(*lightmap, render_scene);
    const int texel_count = static_cast<int>(bake_texels.size());
    if (texel_count == 0) {
        // No valid texels — return empty lightmap.
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        if (settings.lightmap_bake_progress) {
            LightmapBakeProgress& p = *settings.lightmap_bake_progress;
            p.total_texels.store(0, std::memory_order_relaxed);
            p.completed_texels.store(0, std::memory_order_relaxed);
            p.width.store(lightmap->width, std::memory_order_relaxed);
            p.height.store(lightmap->height, std::memory_order_relaxed);
            p.elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
            p.phase.store(static_cast<int>(LightmapBakePhase::Complete), std::memory_order_relaxed);
        }
        return lightmap;
    }

    // 3. Pack and upload the bake scene to GPU.
    PackedGpuScene packed;
    if (!pack_scene_from_render_scene(scene, settings, render_scene, packed)) {
        LT_LOG_ERROR("GPU lightmap bake: failed to pack scene");
        return nullptr;
    }

    void* dev_scene = nullptr;
    void* dev_materials = nullptr;
    void* dev_triangles = nullptr;
    void* dev_traversal_triangles = nullptr;
    void* dev_spheres = nullptr;
    void* dev_triangle_indices = nullptr;
    void* dev_light_indices = nullptr;
    void* dev_directional_lights = nullptr;
    void* dev_point_lights = nullptr;
    void* dev_bvh_nodes = nullptr;
    void* dev_traversal_bvh_nodes = nullptr;
    void* dev_mesh_instances = nullptr;
    void* dev_mesh_instance_indices = nullptr;
    void* dev_tlas_nodes = nullptr;
    void* dev_traversal_tlas_nodes = nullptr;
    void* dev_textures_buf = nullptr;
    int hint_materials = 0, hint_triangles = 0, hint_traversal_triangles = 0, hint_spheres = 0;
    int hint_triangle_indices = 0, hint_light_indices = 0;
    int hint_directional_lights = 0, hint_point_lights = 0, hint_bvh_nodes = 0, hint_traversal_bvh_nodes = 0;
    int hint_mesh_instances = 0, hint_mesh_instance_indices = 0;
    int hint_tlas_nodes = 0, hint_traversal_tlas_nodes = 0, hint_textures = 0;
    std::vector<void*> texture_arrays;
    std::vector<uint64_t> texture_objects;

    cudaError_t cuda_error = cudaSuccess;
    auto cleanup_scene = [&]() {
        cudaFree(dev_scene);
        cudaFree(dev_materials);
        cudaFree(dev_triangles);
        cudaFree(dev_traversal_triangles);
        cudaFree(dev_spheres);
        cudaFree(dev_triangle_indices);
        cudaFree(dev_light_indices);
        cudaFree(dev_directional_lights);
        cudaFree(dev_point_lights);
        cudaFree(dev_bvh_nodes);
        cudaFree(dev_traversal_bvh_nodes);
        cudaFree(dev_mesh_instances);
        cudaFree(dev_mesh_instance_indices);
        cudaFree(dev_tlas_nodes);
        cudaFree(dev_traversal_tlas_nodes);
        cudaFree(dev_textures_buf);
        release_texture_objects(texture_arrays, texture_objects);
    };

    if ((cuda_error = cudaMalloc(&dev_scene, sizeof(GpuScene))) != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: cudaMalloc scene failed: {}", cudaGetErrorString(cuda_error));
        cleanup_scene();
        return nullptr;
    }

    if (!upload_buffer(dev_materials, hint_materials, packed.materials) ||
        !upload_buffer(dev_triangles, hint_triangles, packed.triangles) ||
        !upload_buffer(dev_traversal_triangles, hint_traversal_triangles, packed.traversal_triangles) ||
        !upload_buffer(dev_spheres, hint_spheres, packed.spheres) ||
        !upload_buffer(dev_triangle_indices, hint_triangle_indices, packed.triangle_indices) ||
        !upload_buffer(dev_light_indices, hint_light_indices, packed.light_indices) ||
        !upload_buffer(dev_directional_lights, hint_directional_lights, packed.directional_lights) ||
        !upload_buffer(dev_point_lights, hint_point_lights, packed.point_lights) ||
        !upload_buffer(dev_bvh_nodes, hint_bvh_nodes, packed.bvh_nodes) ||
        !upload_buffer(dev_traversal_bvh_nodes, hint_traversal_bvh_nodes, packed.traversal_bvh_nodes) ||
        !upload_buffer(dev_mesh_instances, hint_mesh_instances, packed.mesh_instances) ||
        !upload_buffer(dev_mesh_instance_indices, hint_mesh_instance_indices, packed.mesh_instance_indices) ||
        !upload_buffer(dev_tlas_nodes, hint_tlas_nodes, packed.tlas_nodes) ||
        !upload_buffer(dev_traversal_tlas_nodes, hint_traversal_tlas_nodes, packed.traversal_tlas_nodes) ||
        !upload_buffer(dev_textures_buf, hint_textures, packed.textures) ||
        !upload_texture_objects(scene, packed, texture_arrays, texture_objects)) {
        LT_LOG_ERROR("GPU lightmap bake: scene upload failed");
        cleanup_scene();
        return nullptr;
    }

    packed.scene.materials = static_cast<GpuMaterial*>(dev_materials);
    packed.scene.textures = static_cast<GpuTexture*>(dev_textures_buf);
    packed.scene.triangles = static_cast<GpuTriangle*>(dev_triangles);
    packed.scene.traversal_triangles = static_cast<GpuTraversalTriangle*>(dev_traversal_triangles);
    packed.scene.spheres = static_cast<GpuSphere*>(dev_spheres);
    packed.scene.triangle_indices = static_cast<int*>(dev_triangle_indices);
    packed.scene.light_indices = static_cast<int*>(dev_light_indices);
    packed.scene.directional_lights = static_cast<GpuDirectionalLight*>(dev_directional_lights);
    packed.scene.point_lights = static_cast<GpuPointLight*>(dev_point_lights);
    packed.scene.bvh_nodes = static_cast<GpuBvhNode*>(dev_bvh_nodes);
    packed.scene.traversal_bvh_nodes = static_cast<GpuTraversalBvhNode*>(dev_traversal_bvh_nodes);
    packed.scene.mesh_instances = static_cast<GpuMeshInstance*>(dev_mesh_instances);
    packed.scene.mesh_instance_indices = static_cast<int*>(dev_mesh_instance_indices);
    packed.scene.tlas_nodes = static_cast<GpuBvhNode*>(dev_tlas_nodes);
    packed.scene.traversal_tlas_nodes = static_cast<GpuTraversalBvhNode*>(dev_traversal_tlas_nodes);

    cuda_error = cudaMemcpy(static_cast<GpuScene*>(dev_scene), &packed.scene,
        sizeof(GpuScene), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: scene header upload failed: {}", cudaGetErrorString(cuda_error));
        cleanup_scene();
        return nullptr;
    }

    // 4. Upload texel records.
    void* dev_bake_texels = nullptr;
    void* dev_lightmap_texels = nullptr;
    void* dev_lightmap_valid = nullptr;
    int hint_bake_texels = 0;
    const size_t lm_size = lightmap->texels.size();
    const size_t texels_bytes = lm_size * sizeof(Vec3);
    const size_t valid_bytes = lm_size * sizeof(uint8_t);

    auto cleanup_all = [&]() {
        cleanup_scene();
        cudaFree(dev_bake_texels);
        cudaFree(dev_lightmap_texels);
        cudaFree(dev_lightmap_valid);
    };

    if (!upload_buffer(dev_bake_texels, hint_bake_texels, bake_texels)) {
        LT_LOG_ERROR("GPU lightmap bake: texel record upload failed");
        cleanup_all();
        return nullptr;
    }

    if ((cuda_error = cudaMalloc(&dev_lightmap_texels, texels_bytes)) != cudaSuccess ||
        (cuda_error = cudaMalloc(&dev_lightmap_valid, valid_bytes)) != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: output buffer alloc failed: {}", cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }
    cuda_error = cudaMemset(dev_lightmap_texels, 0, texels_bytes);
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: memset failed: {}", cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }
    cuda_error = cudaMemset(dev_lightmap_valid, 0, valid_bytes);
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: memset valid failed: {}", cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }

    // 5. Set progress.
    if (settings.lightmap_bake_progress) {
        LightmapBakeProgress& p = *settings.lightmap_bake_progress;
        p.total_texels.store(static_cast<uint64_t>(texel_count), std::memory_order_relaxed);
        p.completed_texels.store(0, std::memory_order_relaxed);
        p.total_rays.store(static_cast<uint64_t>(texel_count) * static_cast<uint64_t>(lightmap->bake_samples) * 2u,
            std::memory_order_relaxed);
        p.width.store(lightmap->width, std::memory_order_relaxed);
        p.height.store(lightmap->height, std::memory_order_relaxed);
        p.phase.store(static_cast<int>(LightmapBakePhase::Baking), std::memory_order_relaxed);
    }

    // 6. Launch bake kernel.
    const int block_size = 256;
    const int grid_size = (texel_count + block_size - 1) / block_size;
    bake_lightmap_texels_kernel<<<grid_size, block_size>>>(
        static_cast<const GpuScene*>(dev_scene),
        static_cast<const GpuLightmapBakeTexel*>(dev_bake_texels),
        texel_count,
        lightmap->bake_samples,
        lightmap->bake_bounces,
        static_cast<int>(settings.sampling_mode),
        static_cast<int>(settings.mis_heuristic),
        static_cast<int>(settings.acceleration_structure),
        std::max(0.0f, settings.emissive_intensity_scale),
        static_cast<Vec3*>(dev_lightmap_texels),
        static_cast<uint8_t*>(dev_lightmap_valid));

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: bake kernel failed: {}", cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }

    // Update progress.
    if (settings.lightmap_bake_progress) {
        LightmapBakeProgress& p = *settings.lightmap_bake_progress;
        p.completed_texels.store(static_cast<uint64_t>(texel_count), std::memory_order_relaxed);
        p.traced_rays.store(static_cast<uint64_t>(texel_count) * static_cast<uint64_t>(lightmap->bake_samples) * 2u,
            std::memory_order_relaxed);
    }

    // 7. Dilation passes (ping-pong with one extra buffer).
    const int dilation_iters = std::clamp(settings.lightmap_dilation, 0, 64);
    if (dilation_iters > 0) {
        void* dev_alt_texels = nullptr;
        void* dev_alt_valid = nullptr;
        if ((cuda_error = cudaMalloc(&dev_alt_texels, texels_bytes)) != cudaSuccess ||
            (cuda_error = cudaMalloc(&dev_alt_valid, valid_bytes)) != cudaSuccess) {
            LT_LOG_ERROR("GPU lightmap bake: dilation buffer alloc failed: {}", cudaGetErrorString(cuda_error));
            cudaFree(dev_alt_texels); cudaFree(dev_alt_valid);
            cleanup_all();
            return nullptr;
        }

        const int lm_grid_size = (static_cast<int>(lm_size) + block_size - 1) / block_size;
        void* src_texels = dev_lightmap_texels;
        void* src_valid = dev_lightmap_valid;
        void* dst_texels = dev_alt_texels;
        void* dst_valid = dev_alt_valid;

        for (int iter = 0; iter < dilation_iters; ++iter) {
            dilate_lightmap_kernel<<<lm_grid_size, block_size>>>(
                static_cast<Vec3*>(src_texels),
                static_cast<uint8_t*>(src_valid),
                lightmap->width,
                lightmap->height,
                static_cast<Vec3*>(dst_texels),
                static_cast<uint8_t*>(dst_valid));

            cuda_error = cudaDeviceSynchronize();
            if (cuda_error != cudaSuccess) {
                LT_LOG_ERROR("GPU lightmap bake: dilation kernel failed at iter {}: {}", iter, cudaGetErrorString(cuda_error));
                cudaFree(dev_alt_texels); cudaFree(dev_alt_valid);
                cleanup_all();
                return nullptr;
            }
            // Swap src and dst for next iteration.
            std::swap(src_texels, dst_texels);
            std::swap(src_valid, dst_valid);
        }
        // After the loop, the final result is in src_texels/src_valid.
        if (src_texels != dev_lightmap_texels) {
            cudaMemcpy(dev_lightmap_texels, src_texels, texels_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_lightmap_valid, src_valid, valid_bytes, cudaMemcpyDeviceToDevice);
        }

        cudaFree(dev_alt_texels); cudaFree(dev_alt_valid);
    }

    // 8. Download results.
    std::vector<Vec3> host_texels(lm_size);
    std::vector<uint8_t> host_valid(lm_size);
    cuda_error = cudaMemcpy(host_texels.data(), dev_lightmap_texels, texels_bytes, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: texel download failed: {}", cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }
    cuda_error = cudaMemcpy(host_valid.data(), dev_lightmap_valid, valid_bytes, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        LT_LOG_ERROR("GPU lightmap bake: valid download failed: {}", cudaGetErrorString(cuda_error));
        cleanup_all();
        return nullptr;
    }

    lightmap->texels = std::move(host_texels);
    lightmap->valid = std::move(host_valid);
    lightmap->baked_texel_count = 0;
    for (uint8_t v : lightmap->valid) {
        if (v) ++lightmap->baked_texel_count;
    }

    // 9. Cleanup temporary device allocations.
    cleanup_all();

    // 10. Final progress and log.
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    if (settings.lightmap_bake_progress) {
        LightmapBakeProgress& p = *settings.lightmap_bake_progress;
        p.elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
        p.phase.store(static_cast<int>(LightmapBakePhase::Complete), std::memory_order_relaxed);
    }

    LT_LOG_INFO("Lightmap baked on GPU: size={}x{} texels={} memory_kib={} elapsed_ms={} elapsed_s={}",
        lightmap->width,
        lightmap->height,
        lightmap->baked_texel_count,
        (lightmap->texels.size() * sizeof(Vec3)) / 1024u,
        format_decimal(elapsed_ms, 3),
        format_decimal(elapsed_ms * 0.001, 2));

    return lightmap;
}

bool CudaPathTracer::available() const {
    configure_cuda_wait_policy_once();
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaPathTracer::~CudaPathTracer() {
    reset();
}

void CudaPathTracer::release_svgf_gbuffer_interop_cache() {
    for (int i = 0; i < 5; ++i) {
        if (svgf_gbuffer_cuda_resources_[i]) {
            cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource_t>(svgf_gbuffer_cuda_resources_[i]));
            svgf_gbuffer_cuda_resources_[i] = nullptr;
        }
        if (svgf_gbuffer_d3d_resources_[i]) {
            static_cast<ID3D11Resource*>(svgf_gbuffer_d3d_resources_[i])->Release();
            svgf_gbuffer_d3d_resources_[i] = nullptr;
        }
    }
    cudaFree(device_svgf_gbuffer_temp_albedo_);
    cudaFree(device_svgf_gbuffer_temp_emission_);
    cudaFree(device_svgf_gbuffer_temp_normal_);
    cudaFree(device_svgf_gbuffer_temp_position_depth_);
    device_svgf_gbuffer_temp_albedo_ = nullptr;
    device_svgf_gbuffer_temp_emission_ = nullptr;
    device_svgf_gbuffer_temp_normal_ = nullptr;
    device_svgf_gbuffer_temp_position_depth_ = nullptr;
    svgf_gbuffer_width_ = 0;
    svgf_gbuffer_height_ = 0;
}

void CudaPathTracer::reset() {
    cudaFree(device_accumulation_);
    cudaFree(device_rgba_);
    cudaFree(device_svgf_radiance_);
    cudaFree(device_svgf_albedo_);
    cudaFree(device_svgf_emission_);
    cudaFree(device_svgf_normal_);
    cudaFree(device_svgf_world_position_);
    cudaFree(device_svgf_depth_);
    cudaFree(device_svgf_object_id_);
    cudaFree(device_svgf_history_illumination_);
    cudaFree(device_svgf_history_moment1_);
    cudaFree(device_svgf_history_moment2_);
    cudaFree(device_svgf_history_length_);
    cudaFree(device_svgf_history_normal_);
    cudaFree(device_svgf_history_world_position_);
    cudaFree(device_svgf_history_depth_);
    cudaFree(device_svgf_history_object_id_);
    cudaFree(device_svgf_temporal_illumination_);
    cudaFree(device_svgf_temporal_moment1_);
    cudaFree(device_svgf_temporal_moment2_);
    cudaFree(device_svgf_temporal_variance_);
    cudaFree(device_svgf_temporal_history_length_);
    cudaFree(device_svgf_ping_illumination_);
    cudaFree(device_svgf_pong_illumination_);
    cudaFree(device_svgf_ping_variance_);
    cudaFree(device_svgf_pong_variance_);
    cudaFree(device_svgf_final_color_);
    cudaFree(device_taa_history_color_);
    cudaFree(device_taa_history_normal_);
    cudaFree(device_taa_history_world_position_);
    cudaFree(device_taa_history_depth_);
    cudaFree(device_taa_history_object_id_);
    cudaFree(device_taa_history_length_);
    release_svgf_gbuffer_interop_cache();
    cudaFree(device_scene_);
    cudaFree(device_materials_);
    cudaFree(device_textures_);
    cudaFree(device_triangles_);
    cudaFree(device_traversal_triangles_);
    cudaFree(device_spheres_);
    cudaFree(device_triangle_indices_);
    cudaFree(device_light_indices_);
    cudaFree(device_directional_lights_);
    cudaFree(device_point_lights_);
    cudaFree(device_bvh_nodes_);
    cudaFree(device_traversal_bvh_nodes_);
    cudaFree(device_traversal_bvh8_nodes_);
    cudaFree(device_traversal_cwbvh_nodes_);
    cudaFree(device_cwbvh_triangle_indices_);
    cudaFree(device_cwbvh_triangles_);
    cudaFree(device_mesh_instances_);
    cudaFree(device_mesh_instance_indices_);
    cudaFree(device_tlas_nodes_);
    cudaFree(device_traversal_tlas_nodes_);
    cudaFree(device_irradiance_volume_directions_);
    cudaFree(device_irradiance_volume_irradiance_);
    cudaFree(device_irradiance_volume_grids_);
    cudaFree(device_irradiance_volume_cells_);
    cudaFree(device_irradiance_volume_debug_probes_);
    cudaFree(device_lightmap_texels_);
    cudaFree(device_wavefront_rays_);
    cudaFree(device_wavefront_throughputs_);
    cudaFree(device_wavefront_radiance_);
    cudaFree(device_wavefront_previous_positions_);
    cudaFree(device_wavefront_previous_bsdf_pdfs_);
    cudaFree(device_wavefront_rngs_);
    cudaFree(device_wavefront_pixels_);
    cudaFree(device_wavefront_states_);
    cudaFree(device_wavefront_compact_hits_);
    cudaFree(device_wavefront_hits_);
    cudaFree(device_wavefront_active_indices_);
    cudaFree(device_wavefront_next_indices_);
    cudaFree(device_wavefront_shade_indices_);
    cudaFree(device_wavefront_shadow_indices_);
    cudaFree(device_wavefront_gi_indices_);
    cudaFree(device_wavefront_bsdf_indices_);
    cudaFree(device_wavefront_queue_counters_);
    cudaFree(device_wavefront_samples_);
    cudaFree(device_wavefront_sample_sum_);
    cudaFree(device_restir_initial_);
    cudaFree(device_restir_temporal_);
    cudaFree(device_restir_history_);
    cudaFree(device_restir_scratch_);
    cudaFree(device_restir_history_surfaces_);
    cudaFree(device_restir_current_surfaces_);
    cudaFree(device_restir_temporal_states_);
    cudaFree(device_restir_spatial_states_);
    cudaFree(device_restir_visibility_rays_);
    cudaFree(device_restir_visibility_results_);
    cudaFree(device_restir_visibility_indices_);
    cudaFree(device_restir_visibility_count_);
    cudaFree(device_restir_neighbor_offsets_);
    cudaFree(device_restir_environment_pmf_);
    cudaFree(device_restir_environment_alias_probability_);
    cudaFree(device_restir_environment_alias_index_);
    cudaFree(device_restir_light_pmf_);
    cudaFree(device_restir_light_alias_probability_);
    cudaFree(device_restir_light_alias_index_);
    cudaFree(device_restir_gi_initial_samples_);
    cudaFree(device_restir_gi_secondary_compact_hits_);
    cudaFree(device_restir_gi_secondary_hits_);
    cudaFree(device_restir_gi_secondary_results_);
    cudaFree(device_restir_gi_secondary_direct_);
    cudaFree(device_restir_gi_initial_);
    cudaFree(device_restir_gi_temporal_);
    cudaFree(device_restir_gi_history_);
    cudaFree(device_restir_gi_scratch_);
    cudaFree(device_restir_gi_current_surfaces_);
    cudaFree(device_restir_gi_history_surfaces_);
    cudaFree(device_restir_gi_temporal_states_);
    cudaFree(device_restir_gi_spatial_states_);
    cudaFree(device_restir_gi_visibility_rays_);
    cudaFree(device_restir_gi_visibility_results_);
    cudaFree(device_restir_pt_path_states_);
    cudaFree(device_restir_pt_compact_hits_);
    cudaFree(device_restir_pt_hits_);
    cudaFree(device_restir_pt_trace_results_);
    cudaFree(device_restir_pt_active_indices_);
    cudaFree(device_restir_pt_next_indices_);
    cudaFree(device_restir_pt_queue_counters_);
    cudaFree(device_restir_pt_nee_reservoirs_);
    cudaFree(device_restir_pt_initial_);
    cudaFree(device_restir_pt_temporal_);
    cudaFree(device_restir_pt_history_);
    cudaFree(device_restir_pt_scratch_);
    cudaFree(device_restir_pt_current_surfaces_);
    cudaFree(device_restir_pt_history_surfaces_);
    cudaFree(device_restir_pt_temporal_states_);
    cudaFree(device_restir_pt_spatial_states_);
    cudaFree(device_restir_pt_visibility_rays_);
    cudaFree(device_restir_pt_visibility_results_);
    cudaFree(device_restir_pt_sample_ids_);
    cudaFree(device_restir_pt_duplication_counts_);
    device_accumulation_ = nullptr;
    device_rgba_ = nullptr;
    device_svgf_radiance_ = nullptr;
    device_svgf_albedo_ = nullptr;
    device_svgf_emission_ = nullptr;
    device_svgf_normal_ = nullptr;
    device_svgf_world_position_ = nullptr;
    device_svgf_depth_ = nullptr;
    device_svgf_object_id_ = nullptr;
    device_svgf_history_illumination_ = nullptr;
    device_svgf_history_moment1_ = nullptr;
    device_svgf_history_moment2_ = nullptr;
    device_svgf_history_length_ = nullptr;
    device_svgf_history_normal_ = nullptr;
    device_svgf_history_world_position_ = nullptr;
    device_svgf_history_depth_ = nullptr;
    device_svgf_history_object_id_ = nullptr;
    device_svgf_temporal_illumination_ = nullptr;
    device_svgf_temporal_moment1_ = nullptr;
    device_svgf_temporal_moment2_ = nullptr;
    device_svgf_temporal_variance_ = nullptr;
    device_svgf_temporal_history_length_ = nullptr;
    device_svgf_ping_illumination_ = nullptr;
    device_svgf_pong_illumination_ = nullptr;
    device_svgf_ping_variance_ = nullptr;
    device_svgf_pong_variance_ = nullptr;
    device_svgf_final_color_ = nullptr;
    device_taa_history_color_ = nullptr;
    device_taa_history_normal_ = nullptr;
    device_taa_history_world_position_ = nullptr;
    device_taa_history_depth_ = nullptr;
    device_taa_history_object_id_ = nullptr;
    device_taa_history_length_ = nullptr;
    device_scene_ = nullptr;
    device_materials_ = nullptr;
    device_textures_ = nullptr;
    device_triangles_ = nullptr;
    device_traversal_triangles_ = nullptr;
    device_spheres_ = nullptr;
    device_triangle_indices_ = nullptr;
    device_light_indices_ = nullptr;
    device_directional_lights_ = nullptr;
    device_point_lights_ = nullptr;
    device_bvh_nodes_ = nullptr;
    device_traversal_bvh_nodes_ = nullptr;
    device_traversal_bvh8_nodes_ = nullptr;
    device_traversal_cwbvh_nodes_ = nullptr;
    device_cwbvh_triangle_indices_ = nullptr;
    device_cwbvh_triangles_ = nullptr;
    device_mesh_instances_ = nullptr;
    device_mesh_instance_indices_ = nullptr;
    device_tlas_nodes_ = nullptr;
    device_traversal_tlas_nodes_ = nullptr;
    device_irradiance_volume_directions_ = nullptr;
    device_irradiance_volume_irradiance_ = nullptr;
    device_irradiance_volume_grids_ = nullptr;
    device_irradiance_volume_cells_ = nullptr;
    device_irradiance_volume_debug_probes_ = nullptr;
    device_lightmap_texels_ = nullptr;
    device_wavefront_rays_ = nullptr;
    device_wavefront_throughputs_ = nullptr;
    device_wavefront_radiance_ = nullptr;
    device_wavefront_previous_positions_ = nullptr;
    device_wavefront_previous_bsdf_pdfs_ = nullptr;
    device_wavefront_rngs_ = nullptr;
    device_wavefront_pixels_ = nullptr;
    device_wavefront_states_ = nullptr;
    device_wavefront_compact_hits_ = nullptr;
    device_wavefront_hits_ = nullptr;
    device_wavefront_active_indices_ = nullptr;
    device_wavefront_next_indices_ = nullptr;
    device_wavefront_shade_indices_ = nullptr;
    device_wavefront_shadow_indices_ = nullptr;
    device_wavefront_gi_indices_ = nullptr;
    device_wavefront_bsdf_indices_ = nullptr;
    device_wavefront_queue_counters_ = nullptr;
    device_wavefront_samples_ = nullptr;
    device_wavefront_sample_sum_ = nullptr;
    device_restir_initial_ = nullptr;
    device_restir_temporal_ = nullptr;
    device_restir_history_ = nullptr;
    device_restir_scratch_ = nullptr;
    device_restir_history_surfaces_ = nullptr;
    device_restir_current_surfaces_ = nullptr;
    device_restir_temporal_states_ = nullptr;
    device_restir_spatial_states_ = nullptr;
    device_restir_visibility_rays_ = nullptr;
    device_restir_visibility_results_ = nullptr;
    device_restir_visibility_indices_ = nullptr;
    device_restir_visibility_count_ = nullptr;
    device_restir_neighbor_offsets_ = nullptr;
    device_restir_environment_pmf_ = nullptr;
    device_restir_environment_alias_probability_ = nullptr;
    device_restir_environment_alias_index_ = nullptr;
    device_restir_light_pmf_ = nullptr;
    device_restir_light_alias_probability_ = nullptr;
    device_restir_light_alias_index_ = nullptr;
    device_restir_gi_initial_samples_ = nullptr;
    device_restir_gi_secondary_compact_hits_ = nullptr;
    device_restir_gi_secondary_hits_ = nullptr;
    device_restir_gi_secondary_results_ = nullptr;
    device_restir_gi_secondary_direct_ = nullptr;
    device_restir_gi_initial_ = nullptr;
    device_restir_gi_temporal_ = nullptr;
    device_restir_gi_history_ = nullptr;
    device_restir_gi_scratch_ = nullptr;
    device_restir_gi_current_surfaces_ = nullptr;
    device_restir_gi_history_surfaces_ = nullptr;
    device_restir_gi_temporal_states_ = nullptr;
    device_restir_gi_spatial_states_ = nullptr;
    device_restir_gi_visibility_rays_ = nullptr;
    device_restir_gi_visibility_results_ = nullptr;
    device_restir_pt_path_states_ = nullptr;
    device_restir_pt_compact_hits_ = nullptr;
    device_restir_pt_hits_ = nullptr;
    device_restir_pt_trace_results_ = nullptr;
    device_restir_pt_active_indices_ = nullptr;
    device_restir_pt_next_indices_ = nullptr;
    device_restir_pt_queue_counters_ = nullptr;
    device_restir_pt_nee_reservoirs_ = nullptr;
    device_restir_pt_initial_ = nullptr;
    device_restir_pt_temporal_ = nullptr;
    device_restir_pt_history_ = nullptr;
    device_restir_pt_scratch_ = nullptr;
    device_restir_pt_current_surfaces_ = nullptr;
    device_restir_pt_history_surfaces_ = nullptr;
    device_restir_pt_temporal_states_ = nullptr;
    device_restir_pt_spatial_states_ = nullptr;
    device_restir_pt_visibility_rays_ = nullptr;
    device_restir_pt_visibility_results_ = nullptr;
    device_restir_pt_sample_ids_ = nullptr;
    device_restir_pt_duplication_counts_ = nullptr;
    cached_render_scene_ = {};
    cached_irradiance_volume_.reset();
    cached_lightmap_.reset();
    cached_pixels_ = 0;
    cached_materials_ = 0;
    cached_textures_ = 0;
    cached_triangles_ = 0;
    cached_traversal_triangles_ = 0;
    cached_spheres_ = 0;
    cached_triangle_indices_ = 0;
    cached_lights_ = 0;
    cached_directional_lights_ = 0;
    cached_point_lights_ = 0;
    cached_bvh_nodes_ = 0;
    cached_traversal_bvh_nodes_ = 0;
    cached_traversal_bvh8_nodes_ = 0;
    cached_traversal_cwbvh_nodes_ = 0;
    cached_cwbvh_triangle_indices_ = 0;
    cached_cwbvh_triangles_ = 0;
    cached_mesh_instances_ = 0;
    cached_mesh_instance_indices_ = 0;
    cached_tlas_nodes_ = 0;
    cached_traversal_tlas_nodes_ = 0;
    cached_irradiance_volume_directions_ = 0;
    cached_irradiance_volume_irradiance_ = 0;
    cached_irradiance_volume_grids_ = 0;
    cached_irradiance_volume_cells_ = 0;
    cached_irradiance_volume_debug_probes_ = 0;
    cached_lightmap_texels_ = 0;
    scene_uploaded_ = false;
    cached_render_scene_valid_ = false;
    cached_irradiance_volume_enabled_ = false;
    cached_lightmap_enabled_ = false;
    svgf_history_valid_ = false;
    svgf_history_camera_ = {};
    svgf_history_jitter_x_ = 0.0f;
    svgf_history_jitter_y_ = 0.0f;
    taa_history_valid_ = false;
    taa_history_camera_ = {};
    taa_history_jitter_x_ = 0.0f;
    taa_history_jitter_y_ = 0.0f;
    restir_history_valid_ = false;
    restir_was_enabled_ = false;
    restir_history_camera_ = {};
    cached_restir_environment_texels_ = 0;
    cached_restir_light_count_ = 0;
    restir_history_bias_correction_ = RestirBiasCorrection::Basic;
    restir_history_visibility_reuse_ = false;
    restir_gi_history_valid_ = false;
    restir_gi_was_enabled_ = false;
    restir_gi_history_camera_ = {};
    restir_gi_history_bias_correction_ = RestirBiasCorrection::Basic;
    restir_gi_history_resampling_ = RestirGiResamplingMode::TemporalSpatial;
    restir_gi_history_final_mis_ = true;
    restir_gi_history_boiling_filter_ = true;
    restir_gi_history_secondary_roughness_ = 0.5f;
    restir_pt_history_valid_ = false;
    restir_pt_was_enabled_ = false;
    restir_pt_history_camera_ = {};
    restir_pt_history_resampling_ = RestirPtResamplingMode::TemporalSpatial;
    restir_pt_history_reconnection_ = RestirPtReconnectionMode::Footprint;
    restir_pt_history_max_bounces_ = 3;
    release_texture_objects(texture_arrays_, texture_objects_);
}

void CudaPathTracer::render_cpu_fallback(
    const Scene& scene,
    const RenderSettings& settings,
    Framebuffer& framebuffer,
    const char* reason,
    const char* detail) {
    std::string key = reason ? reason : "unknown CUDA failure";
    if (detail && *detail) {
        key += ": ";
        key += detail;
    }
    if (std::find(reported_fallback_reasons_.begin(), reported_fallback_reasons_.end(), key) == reported_fallback_reasons_.end()) {
        reported_fallback_reasons_.push_back(key);
        LT_LOG_WARN("CUDA path tracer fallback to CPU: {}", key);
    } else {
        LT_LOG_DEBUG("CUDA path tracer fallback to CPU: {}", key);
    }
    CpuPathTracer fallback;
    fallback.render(scene, settings, framebuffer);
}

void CudaPathTracer::render_with_rasterized_gbuffer_interop(
    const Scene& scene,
    const RenderSettings& settings,
    Framebuffer& framebuffer,
    const RasterizedGBufferInterop& interop) {
    svgf_gbuffer_interop_ = interop;
    render(scene, settings, framebuffer);
    svgf_gbuffer_interop_ = {};
}

bool CudaPathTracer::upload_rasterized_gbuffer_interop_to_cuda(
    const RasterizedGBufferInterop& interop,
    int width,
    int height,
    Vec3* albedo,
    Vec3* emission,
    Vec3* normal,
    Vec3* world_position,
    float* depth,
    uint32_t* object_id,
    int& cuda_error_code) {
    cudaError_t cuda_error = cudaSuccess;
    if (!interop.valid ||
        interop.width != width ||
        interop.height != height ||
        !interop.albedo_texture ||
        !interop.emission_texture ||
        !interop.normal_texture ||
        !interop.position_depth_texture ||
        !interop.object_id_texture) {
        cuda_error = cudaErrorInvalidValue;
        cuda_error_code = static_cast<int>(cuda_error);
        return false;
    }

    void* textures[5] = {
        interop.albedo_texture,
        interop.emission_texture,
        interop.normal_texture,
        interop.position_depth_texture,
        interop.object_id_texture,
    };
    bool cached_resources_match = svgf_gbuffer_width_ == width && svgf_gbuffer_height_ == height;
    for (int i = 0; i < 5; ++i) {
        cached_resources_match = cached_resources_match &&
            svgf_gbuffer_cuda_resources_[i] &&
            svgf_gbuffer_d3d_resources_[i] == textures[i];
    }
    if (!cached_resources_match) {
        release_svgf_gbuffer_interop_cache();
        for (int i = 0; i < 5; ++i) {
            static_cast<ID3D11Resource*>(textures[i])->AddRef();
            svgf_gbuffer_d3d_resources_[i] = textures[i];
            cuda_error = cudaGraphicsD3D11RegisterResource(
                reinterpret_cast<cudaGraphicsResource_t*>(&svgf_gbuffer_cuda_resources_[i]),
                static_cast<ID3D11Resource*>(textures[i]),
                cudaGraphicsRegisterFlagsReadOnly);
            if (cuda_error != cudaSuccess) {
                release_svgf_gbuffer_interop_cache();
                cuda_error_code = static_cast<int>(cuda_error);
                return false;
            }
        }
        svgf_gbuffer_width_ = width;
        svgf_gbuffer_height_ = height;
    }

    cudaGraphicsResource_t resources[5] = {};
    for (int i = 0; i < 5; ++i) {
        resources[i] = static_cast<cudaGraphicsResource_t>(svgf_gbuffer_cuda_resources_[i]);
    }
    bool mapped = false;

    const auto cleanup = [&]() {
        if (mapped) {
            const cudaError_t unmap_error = cudaGraphicsUnmapResources(5, resources, 0);
            if (cuda_error == cudaSuccess && unmap_error != cudaSuccess) {
                cuda_error = unmap_error;
            }
            mapped = false;
        }
    };

    cuda_error = cudaGraphicsMapResources(5, resources, 0);
    if (cuda_error != cudaSuccess) {
        cleanup();
        release_svgf_gbuffer_interop_cache();
        cuda_error_code = static_cast<int>(cuda_error);
        return false;
    }
    mapped = true;

    cudaArray_t arrays[5] = {};
    for (int i = 0; i < 5; ++i) {
        cuda_error = cudaGraphicsSubResourceGetMappedArray(&arrays[i], resources[i], 0, 0);
        if (cuda_error != cudaSuccess) {
            cleanup();
            release_svgf_gbuffer_interop_cache();
            cuda_error_code = static_cast<int>(cuda_error);
            return false;
        }
    }

    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t float4_pitch = static_cast<size_t>(width) * sizeof(GBufferFloat4);
    const size_t object_id_pitch = static_cast<size_t>(width) * sizeof(uint32_t);
    if (!device_svgf_gbuffer_temp_albedo_) cuda_error = cudaMalloc(&device_svgf_gbuffer_temp_albedo_, pixels * sizeof(GBufferFloat4));
    if (cuda_error == cudaSuccess && !device_svgf_gbuffer_temp_emission_) cuda_error = cudaMalloc(&device_svgf_gbuffer_temp_emission_, pixels * sizeof(GBufferFloat4));
    if (cuda_error == cudaSuccess && !device_svgf_gbuffer_temp_normal_) cuda_error = cudaMalloc(&device_svgf_gbuffer_temp_normal_, pixels * sizeof(GBufferFloat4));
    if (cuda_error == cudaSuccess && !device_svgf_gbuffer_temp_position_depth_) cuda_error = cudaMalloc(&device_svgf_gbuffer_temp_position_depth_, pixels * sizeof(GBufferFloat4));
    if (cuda_error != cudaSuccess) {
        cleanup();
        release_svgf_gbuffer_interop_cache();
        cuda_error_code = static_cast<int>(cuda_error);
        return false;
    }

    GBufferFloat4* temp_albedo = static_cast<GBufferFloat4*>(device_svgf_gbuffer_temp_albedo_);
    GBufferFloat4* temp_emission = static_cast<GBufferFloat4*>(device_svgf_gbuffer_temp_emission_);
    GBufferFloat4* temp_normal = static_cast<GBufferFloat4*>(device_svgf_gbuffer_temp_normal_);
    GBufferFloat4* temp_position_depth = static_cast<GBufferFloat4*>(device_svgf_gbuffer_temp_position_depth_);
    cuda_error = cudaMemcpy2DFromArray(temp_albedo, float4_pitch, arrays[0], 0, 0, float4_pitch, height, cudaMemcpyDeviceToDevice);
    if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpy2DFromArray(temp_emission, float4_pitch, arrays[1], 0, 0, float4_pitch, height, cudaMemcpyDeviceToDevice);
    }
    if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpy2DFromArray(temp_normal, float4_pitch, arrays[2], 0, 0, float4_pitch, height, cudaMemcpyDeviceToDevice);
    }
    if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpy2DFromArray(temp_position_depth, float4_pitch, arrays[3], 0, 0, float4_pitch, height, cudaMemcpyDeviceToDevice);
    }
    if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpy2DFromArray(object_id, object_id_pitch, arrays[4], 0, 0, object_id_pitch, height, cudaMemcpyDeviceToDevice);
    }
    if (cuda_error != cudaSuccess) {
        cleanup();
        release_svgf_gbuffer_interop_cache();
        cuda_error_code = static_cast<int>(cuda_error);
        return false;
    }

    const int count = static_cast<int>(pixels);
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    unpack_rasterized_gbuffer_kernel<<<grid_size, block_size>>>(
        temp_albedo,
        temp_emission,
        temp_normal,
        temp_position_depth,
        albedo,
        emission,
        normal,
        world_position,
        depth,
        count);
    cuda_error = cudaDeviceSynchronize();
    cleanup();
    if (cuda_error != cudaSuccess) {
        release_svgf_gbuffer_interop_cache();
    }
    cuda_error_code = static_cast<int>(cuda_error);
    return cuda_error == cudaSuccess;
}

void CudaPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    const ScopedNvtxRange frame_range("LightTransport CUDA frame");
    configure_cuda_wait_policy_once();
    framebuffer.resize(settings.width, settings.height);
    const size_t pixels = static_cast<size_t>(settings.width) * static_cast<size_t>(settings.height);

    if (cached_pixels_ != pixels && cached_pixels_ != 0) {
        cudaFree(device_accumulation_);
        cudaFree(device_rgba_);
        cudaFree(device_svgf_radiance_);
        cudaFree(device_svgf_albedo_);
        cudaFree(device_svgf_emission_);
        cudaFree(device_svgf_normal_);
        cudaFree(device_svgf_world_position_);
        cudaFree(device_svgf_depth_);
        cudaFree(device_svgf_object_id_);
        cudaFree(device_svgf_history_illumination_);
        cudaFree(device_svgf_history_moment1_);
        cudaFree(device_svgf_history_moment2_);
        cudaFree(device_svgf_history_length_);
        cudaFree(device_svgf_history_normal_);
        cudaFree(device_svgf_history_world_position_);
        cudaFree(device_svgf_history_depth_);
        cudaFree(device_svgf_history_object_id_);
        cudaFree(device_svgf_temporal_illumination_);
        cudaFree(device_svgf_temporal_moment1_);
        cudaFree(device_svgf_temporal_moment2_);
        cudaFree(device_svgf_temporal_variance_);
        cudaFree(device_svgf_temporal_history_length_);
        cudaFree(device_svgf_ping_illumination_);
        cudaFree(device_svgf_pong_illumination_);
        cudaFree(device_svgf_ping_variance_);
        cudaFree(device_svgf_pong_variance_);
        cudaFree(device_svgf_final_color_);
        cudaFree(device_taa_history_color_);
        cudaFree(device_taa_history_normal_);
        cudaFree(device_taa_history_world_position_);
        cudaFree(device_taa_history_depth_);
        cudaFree(device_taa_history_object_id_);
        cudaFree(device_taa_history_length_);
        cudaFree(device_svgf_gbuffer_temp_albedo_);
        cudaFree(device_svgf_gbuffer_temp_emission_);
        cudaFree(device_svgf_gbuffer_temp_normal_);
        cudaFree(device_svgf_gbuffer_temp_position_depth_);
        cudaFree(device_wavefront_rays_);
        cudaFree(device_wavefront_throughputs_);
        cudaFree(device_wavefront_radiance_);
        cudaFree(device_wavefront_previous_positions_);
        cudaFree(device_wavefront_previous_bsdf_pdfs_);
        cudaFree(device_wavefront_rngs_);
        cudaFree(device_wavefront_pixels_);
        cudaFree(device_wavefront_states_);
        cudaFree(device_wavefront_compact_hits_);
        cudaFree(device_wavefront_hits_);
        cudaFree(device_wavefront_active_indices_);
        cudaFree(device_wavefront_next_indices_);
        cudaFree(device_wavefront_shade_indices_);
        cudaFree(device_wavefront_shadow_indices_);
        cudaFree(device_wavefront_gi_indices_);
        cudaFree(device_wavefront_bsdf_indices_);
        cudaFree(device_wavefront_queue_counters_);
        cudaFree(device_wavefront_samples_);
        cudaFree(device_wavefront_sample_sum_);
        cudaFree(device_restir_initial_);
        cudaFree(device_restir_temporal_);
        cudaFree(device_restir_history_);
        cudaFree(device_restir_scratch_);
        cudaFree(device_restir_history_surfaces_);
        cudaFree(device_restir_current_surfaces_);
        cudaFree(device_restir_temporal_states_);
        cudaFree(device_restir_spatial_states_);
        cudaFree(device_restir_visibility_rays_);
        cudaFree(device_restir_visibility_results_);
        cudaFree(device_restir_visibility_indices_);
        cudaFree(device_restir_visibility_count_);
        cudaFree(device_restir_neighbor_offsets_);
        cudaFree(device_restir_environment_pmf_);
        cudaFree(device_restir_environment_alias_probability_);
        cudaFree(device_restir_environment_alias_index_);
        cudaFree(device_restir_light_pmf_);
        cudaFree(device_restir_light_alias_probability_);
        cudaFree(device_restir_light_alias_index_);
        cudaFree(device_restir_gi_initial_samples_);
        cudaFree(device_restir_gi_secondary_compact_hits_);
        cudaFree(device_restir_gi_secondary_hits_);
        cudaFree(device_restir_gi_secondary_results_);
        cudaFree(device_restir_gi_secondary_direct_);
        cudaFree(device_restir_gi_initial_);
        cudaFree(device_restir_gi_temporal_);
        cudaFree(device_restir_gi_history_);
        cudaFree(device_restir_gi_scratch_);
        cudaFree(device_restir_gi_current_surfaces_);
        cudaFree(device_restir_gi_history_surfaces_);
        cudaFree(device_restir_gi_temporal_states_);
        cudaFree(device_restir_gi_spatial_states_);
        cudaFree(device_restir_gi_visibility_rays_);
        cudaFree(device_restir_gi_visibility_results_);
        cudaFree(device_restir_pt_path_states_);
        cudaFree(device_restir_pt_compact_hits_);
        cudaFree(device_restir_pt_hits_);
        cudaFree(device_restir_pt_trace_results_);
        cudaFree(device_restir_pt_active_indices_);
        cudaFree(device_restir_pt_next_indices_);
        cudaFree(device_restir_pt_queue_counters_);
        cudaFree(device_restir_pt_nee_reservoirs_);
        cudaFree(device_restir_pt_initial_);
        cudaFree(device_restir_pt_temporal_);
        cudaFree(device_restir_pt_history_);
        cudaFree(device_restir_pt_scratch_);
        cudaFree(device_restir_pt_current_surfaces_);
        cudaFree(device_restir_pt_history_surfaces_);
        cudaFree(device_restir_pt_temporal_states_);
        cudaFree(device_restir_pt_spatial_states_);
        cudaFree(device_restir_pt_visibility_rays_);
        cudaFree(device_restir_pt_visibility_results_);
        cudaFree(device_restir_pt_sample_ids_);
        cudaFree(device_restir_pt_duplication_counts_);

        device_accumulation_ = nullptr;
        device_rgba_ = nullptr;
        device_svgf_radiance_ = nullptr;
        device_svgf_albedo_ = nullptr;
        device_svgf_emission_ = nullptr;
        device_svgf_normal_ = nullptr;
        device_svgf_world_position_ = nullptr;
        device_svgf_depth_ = nullptr;
        device_svgf_object_id_ = nullptr;
        device_svgf_history_illumination_ = nullptr;
        device_svgf_history_moment1_ = nullptr;
        device_svgf_history_moment2_ = nullptr;
        device_svgf_history_length_ = nullptr;
        device_svgf_history_normal_ = nullptr;
        device_svgf_history_world_position_ = nullptr;
        device_svgf_history_depth_ = nullptr;
        device_svgf_history_object_id_ = nullptr;
        device_svgf_temporal_illumination_ = nullptr;
        device_svgf_temporal_moment1_ = nullptr;
        device_svgf_temporal_moment2_ = nullptr;
        device_svgf_temporal_variance_ = nullptr;
        device_svgf_temporal_history_length_ = nullptr;
        device_svgf_ping_illumination_ = nullptr;
        device_svgf_pong_illumination_ = nullptr;
        device_svgf_ping_variance_ = nullptr;
        device_svgf_pong_variance_ = nullptr;
        device_svgf_final_color_ = nullptr;
        device_taa_history_color_ = nullptr;
        device_taa_history_normal_ = nullptr;
        device_taa_history_world_position_ = nullptr;
        device_taa_history_depth_ = nullptr;
        device_taa_history_object_id_ = nullptr;
        device_taa_history_length_ = nullptr;
        device_svgf_gbuffer_temp_albedo_ = nullptr;
        device_svgf_gbuffer_temp_emission_ = nullptr;
        device_svgf_gbuffer_temp_normal_ = nullptr;
        device_svgf_gbuffer_temp_position_depth_ = nullptr;
        device_wavefront_rays_ = nullptr;
        device_wavefront_throughputs_ = nullptr;
        device_wavefront_radiance_ = nullptr;
        device_wavefront_previous_positions_ = nullptr;
        device_wavefront_previous_bsdf_pdfs_ = nullptr;
        device_wavefront_rngs_ = nullptr;
        device_wavefront_pixels_ = nullptr;
        device_wavefront_states_ = nullptr;
        device_wavefront_compact_hits_ = nullptr;
        device_wavefront_hits_ = nullptr;
        device_wavefront_active_indices_ = nullptr;
        device_wavefront_next_indices_ = nullptr;
        device_wavefront_shade_indices_ = nullptr;
        device_wavefront_shadow_indices_ = nullptr;
        device_wavefront_gi_indices_ = nullptr;
        device_wavefront_bsdf_indices_ = nullptr;
        device_wavefront_queue_counters_ = nullptr;
        device_wavefront_samples_ = nullptr;
        device_wavefront_sample_sum_ = nullptr;
        device_restir_initial_ = nullptr;
        device_restir_temporal_ = nullptr;
        device_restir_history_ = nullptr;
        device_restir_scratch_ = nullptr;
        device_restir_history_surfaces_ = nullptr;
        device_restir_current_surfaces_ = nullptr;
        device_restir_temporal_states_ = nullptr;
        device_restir_spatial_states_ = nullptr;
        device_restir_visibility_rays_ = nullptr;
        device_restir_visibility_results_ = nullptr;
        device_restir_visibility_indices_ = nullptr;
        device_restir_visibility_count_ = nullptr;
        device_restir_neighbor_offsets_ = nullptr;
        device_restir_environment_pmf_ = nullptr;
        device_restir_environment_alias_probability_ = nullptr;
        device_restir_environment_alias_index_ = nullptr;
        device_restir_light_pmf_ = nullptr;
        device_restir_light_alias_probability_ = nullptr;
        device_restir_light_alias_index_ = nullptr;
        device_restir_gi_initial_samples_ = nullptr;
        device_restir_gi_secondary_compact_hits_ = nullptr;
        device_restir_gi_secondary_hits_ = nullptr;
        device_restir_gi_secondary_results_ = nullptr;
        device_restir_gi_secondary_direct_ = nullptr;
        device_restir_gi_initial_ = nullptr;
        device_restir_gi_temporal_ = nullptr;
        device_restir_gi_history_ = nullptr;
        device_restir_gi_scratch_ = nullptr;
        device_restir_gi_current_surfaces_ = nullptr;
        device_restir_gi_history_surfaces_ = nullptr;
        device_restir_gi_temporal_states_ = nullptr;
        device_restir_gi_spatial_states_ = nullptr;
        device_restir_gi_visibility_rays_ = nullptr;
        device_restir_gi_visibility_results_ = nullptr;
        device_restir_pt_path_states_ = nullptr;
        device_restir_pt_compact_hits_ = nullptr;
        device_restir_pt_hits_ = nullptr;
        device_restir_pt_trace_results_ = nullptr;
        device_restir_pt_active_indices_ = nullptr;
        device_restir_pt_next_indices_ = nullptr;
        device_restir_pt_queue_counters_ = nullptr;
        device_restir_pt_nee_reservoirs_ = nullptr;
        device_restir_pt_initial_ = nullptr;
        device_restir_pt_temporal_ = nullptr;
        device_restir_pt_history_ = nullptr;
        device_restir_pt_scratch_ = nullptr;
        device_restir_pt_current_surfaces_ = nullptr;
        device_restir_pt_history_surfaces_ = nullptr;
        device_restir_pt_temporal_states_ = nullptr;
        device_restir_pt_spatial_states_ = nullptr;
        device_restir_pt_visibility_rays_ = nullptr;
        device_restir_pt_visibility_results_ = nullptr;
        device_restir_pt_sample_ids_ = nullptr;
        device_restir_pt_duplication_counts_ = nullptr;
        cached_pixels_ = 0;
        svgf_history_valid_ = false;
        taa_history_valid_ = false;
        restir_history_valid_ = false;
        restir_gi_history_valid_ = false;
        restir_pt_history_valid_ = false;
        release_svgf_gbuffer_interop_cache();
    }

    cudaError_t cuda_error = cudaSuccess;
    if (!device_accumulation_ && (cuda_error = cudaMalloc(&device_accumulation_, pixels * sizeof(Vec3))) != cudaSuccess) {
        reset();
        render_cpu_fallback(scene, settings, framebuffer, "could not allocate accumulation buffer", cuda_error_text(cuda_error));
        return;
    }
    if (!device_rgba_ && (cuda_error = cudaMalloc(&device_rgba_, pixels * sizeof(uint32_t))) != cudaSuccess) {
        reset();
        render_cpu_fallback(scene, settings, framebuffer, "could not allocate RGBA buffer", cuda_error_text(cuda_error));
        return;
    }
    if (!device_scene_ && (cuda_error = cudaMalloc(&device_scene_, sizeof(GpuScene))) != cudaSuccess) {
        reset();
        render_cpu_fallback(scene, settings, framebuffer, "could not allocate scene buffer", cuda_error_text(cuda_error));
        return;
    }
    const auto allocate_svgf_buffer = [&](void*& ptr, size_t bytes, const char* name) {
        if (ptr) {
            return true;
        }
        cuda_error = cudaMalloc(&ptr, bytes);
        if (cuda_error != cudaSuccess) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, name, cuda_error_text(cuda_error));
            return false;
        }
        return true;
    };
    if (svgf_denoising_enabled(settings)) {
        if (!allocate_svgf_buffer(device_svgf_radiance_, pixels * sizeof(Vec3), "could not allocate SVGF radiance buffer") ||
            !allocate_svgf_buffer(device_svgf_albedo_, pixels * sizeof(Vec3), "could not allocate SVGF albedo buffer") ||
            !allocate_svgf_buffer(device_svgf_emission_, pixels * sizeof(Vec3), "could not allocate SVGF emission buffer") ||
            !allocate_svgf_buffer(device_svgf_normal_, pixels * sizeof(Vec3), "could not allocate SVGF normal buffer") ||
            !allocate_svgf_buffer(device_svgf_world_position_, pixels * sizeof(Vec3), "could not allocate SVGF position buffer") ||
            !allocate_svgf_buffer(device_svgf_depth_, pixels * sizeof(float), "could not allocate SVGF depth buffer") ||
            !allocate_svgf_buffer(device_svgf_object_id_, pixels * sizeof(uint32_t), "could not allocate SVGF object id buffer") ||
            !allocate_svgf_buffer(device_svgf_history_illumination_, pixels * sizeof(Vec3), "could not allocate SVGF history illumination buffer") ||
            !allocate_svgf_buffer(device_svgf_history_moment1_, pixels * sizeof(float), "could not allocate SVGF history moment buffer") ||
            !allocate_svgf_buffer(device_svgf_history_moment2_, pixels * sizeof(float), "could not allocate SVGF history moment buffer") ||
            !allocate_svgf_buffer(device_svgf_history_length_, pixels * sizeof(float), "could not allocate SVGF history length buffer") ||
            !allocate_svgf_buffer(device_svgf_history_normal_, pixels * sizeof(Vec3), "could not allocate SVGF history normal buffer") ||
            !allocate_svgf_buffer(device_svgf_history_world_position_, pixels * sizeof(Vec3), "could not allocate SVGF history position buffer") ||
            !allocate_svgf_buffer(device_svgf_history_depth_, pixels * sizeof(float), "could not allocate SVGF history depth buffer") ||
            !allocate_svgf_buffer(device_svgf_history_object_id_, pixels * sizeof(uint32_t), "could not allocate SVGF history object id buffer") ||
            !allocate_svgf_buffer(device_svgf_temporal_illumination_, pixels * sizeof(Vec3), "could not allocate SVGF temporal illumination buffer") ||
            !allocate_svgf_buffer(device_svgf_temporal_moment1_, pixels * sizeof(float), "could not allocate SVGF temporal moment buffer") ||
            !allocate_svgf_buffer(device_svgf_temporal_moment2_, pixels * sizeof(float), "could not allocate SVGF temporal moment buffer") ||
            !allocate_svgf_buffer(device_svgf_temporal_variance_, pixels * sizeof(float), "could not allocate SVGF temporal variance buffer") ||
            !allocate_svgf_buffer(device_svgf_temporal_history_length_, pixels * sizeof(float), "could not allocate SVGF temporal history buffer") ||
            !allocate_svgf_buffer(device_svgf_ping_illumination_, pixels * sizeof(Vec3), "could not allocate SVGF ping illumination buffer") ||
            !allocate_svgf_buffer(device_svgf_pong_illumination_, pixels * sizeof(Vec3), "could not allocate SVGF pong illumination buffer") ||
            !allocate_svgf_buffer(device_svgf_ping_variance_, pixels * sizeof(float), "could not allocate SVGF ping variance buffer") ||
            !allocate_svgf_buffer(device_svgf_pong_variance_, pixels * sizeof(float), "could not allocate SVGF pong variance buffer") ||
            !allocate_svgf_buffer(device_svgf_final_color_, pixels * sizeof(Vec3), "could not allocate SVGF final color buffer") ||
            !allocate_svgf_buffer(device_taa_history_color_, pixels * sizeof(Vec3), "could not allocate TAA history color buffer") ||
            !allocate_svgf_buffer(device_taa_history_normal_, pixels * sizeof(Vec3), "could not allocate TAA history normal buffer") ||
            !allocate_svgf_buffer(device_taa_history_world_position_, pixels * sizeof(Vec3), "could not allocate TAA history position buffer") ||
            !allocate_svgf_buffer(device_taa_history_depth_, pixels * sizeof(float), "could not allocate TAA history depth buffer") ||
            !allocate_svgf_buffer(device_taa_history_object_id_, pixels * sizeof(uint32_t), "could not allocate TAA history object id buffer") ||
            !allocate_svgf_buffer(device_taa_history_length_, pixels * sizeof(float), "could not allocate TAA history length buffer")) {
            return;
        }
    }
    const bool wavefront_enabled = settings.cuda_wavefront &&
        !(settings.use_irradiance_volume && settings.irradiance_volume_debug_probes);
    const bool restir_enabled = wavefront_enabled && settings.cuda_restir_di &&
        settings.sampling_mode != PathSamplingMode::Unidirectional;
    const bool restir_pt_enabled = wavefront_enabled && settings.cuda_restir_pt &&
        settings.sampling_mode != PathSamplingMode::Unidirectional &&
        settings.max_bounces >= 2 &&
        !settings.use_lightmap && !settings.use_irradiance_volume;
    const bool restir_gi_enabled = wavefront_enabled && settings.cuda_restir_gi &&
        settings.sampling_mode != PathSamplingMode::Unidirectional &&
        settings.max_bounces >= 2 &&
        !settings.use_lightmap && !settings.use_irradiance_volume && !restir_pt_enabled;
    const bool restir_any_enabled = restir_enabled || restir_gi_enabled || restir_pt_enabled;
    if (settings.cuda_restir_gi && settings.cuda_restir_pt) {
        const std::string key = "ReSTIR GI and PT requested together; PT takes precedence";
        if (std::find(reported_fallback_reasons_.begin(), reported_fallback_reasons_.end(), key) ==
            reported_fallback_reasons_.end()) {
            reported_fallback_reasons_.push_back(key);
            LT_LOG_WARN("{}", key);
        }
    }
    const bool upload_restir_neighbor_offsets = restir_enabled && !device_restir_neighbor_offsets_;
    if (wavefront_enabled) {
        if (!allocate_svgf_buffer(device_wavefront_rays_, pixels * sizeof(Ray), "could not allocate wavefront ray buffer") ||
            !allocate_svgf_buffer(device_wavefront_throughputs_, pixels * sizeof(Vec3), "could not allocate wavefront throughput buffer") ||
            !allocate_svgf_buffer(device_wavefront_radiance_, pixels * sizeof(Vec3), "could not allocate wavefront radiance buffer") ||
            !allocate_svgf_buffer(device_wavefront_previous_positions_, pixels * sizeof(Vec3), "could not allocate wavefront previous-position buffer") ||
            !allocate_svgf_buffer(device_wavefront_previous_bsdf_pdfs_, pixels * sizeof(float), "could not allocate wavefront previous-pdf buffer") ||
            !allocate_svgf_buffer(device_wavefront_rngs_, pixels * sizeof(uint32_t), "could not allocate wavefront rng buffer") ||
            !allocate_svgf_buffer(device_wavefront_pixels_, pixels * sizeof(int), "could not allocate wavefront pixel buffer") ||
            !allocate_svgf_buffer(device_wavefront_states_, pixels * sizeof(uint32_t), "could not allocate wavefront state buffer") ||
            !allocate_svgf_buffer(device_wavefront_compact_hits_, pixels * sizeof(GpuCompactHit), "could not allocate wavefront compact hit buffer") ||
            !allocate_svgf_buffer(device_wavefront_hits_, pixels * sizeof(GpuHit), "could not allocate wavefront hit buffer") ||
            !allocate_svgf_buffer(device_wavefront_active_indices_, pixels * sizeof(int), "could not allocate wavefront active queue") ||
            !allocate_svgf_buffer(device_wavefront_next_indices_, pixels * sizeof(int), "could not allocate wavefront next queue") ||
            !allocate_svgf_buffer(device_wavefront_shade_indices_, pixels * sizeof(int), "could not allocate wavefront shade queue") ||
            !allocate_svgf_buffer(device_wavefront_shadow_indices_, pixels * sizeof(int), "could not allocate wavefront shadow queue") ||
            !allocate_svgf_buffer(device_wavefront_gi_indices_, pixels * sizeof(int), "could not allocate wavefront GI queue") ||
            !allocate_svgf_buffer(device_wavefront_bsdf_indices_, pixels * sizeof(int), "could not allocate wavefront bsdf queue") ||
            !allocate_svgf_buffer(device_wavefront_queue_counters_, sizeof(GpuWavefrontQueueCounters), "could not allocate wavefront queue counters") ||
            !allocate_svgf_buffer(device_wavefront_samples_, pixels * sizeof(Vec3), "could not allocate wavefront sample buffer") ||
            !allocate_svgf_buffer(device_wavefront_sample_sum_, pixels * sizeof(Vec3), "could not allocate wavefront sample sum buffer")) {
            return;
        }
        if (restir_enabled &&
            (!allocate_svgf_buffer(device_restir_initial_, pixels * sizeof(GpuRestirReservoir), "could not allocate ReSTIR initial reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_temporal_, pixels * sizeof(GpuRestirReservoir), "could not allocate ReSTIR temporal reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_history_, pixels * sizeof(GpuRestirReservoir), "could not allocate ReSTIR history reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_scratch_, pixels * sizeof(GpuRestirReservoir), "could not allocate ReSTIR spatial reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_history_surfaces_, pixels * sizeof(GpuRestirSurface), "could not allocate ReSTIR history surface buffer") ||
             !allocate_svgf_buffer(device_restir_current_surfaces_, pixels * sizeof(GpuRestirSurface), "could not allocate ReSTIR current surface buffer") ||
             !allocate_svgf_buffer(device_restir_temporal_states_, pixels * sizeof(GpuRestirTemporalState), "could not allocate ReSTIR temporal state buffer") ||
             !allocate_svgf_buffer(device_restir_spatial_states_, pixels * sizeof(GpuRestirSpatialState), "could not allocate ReSTIR spatial state buffer") ||
             !allocate_svgf_buffer(device_restir_visibility_rays_, pixels * sizeof(GpuRestirVisibilityRay), "could not allocate ReSTIR visibility ray buffer") ||
             !allocate_svgf_buffer(device_restir_visibility_results_, pixels * sizeof(int), "could not allocate ReSTIR visibility result buffer") ||
             !allocate_svgf_buffer(device_restir_neighbor_offsets_, kRestirNeighborOffsetCount * sizeof(Vec2), "could not allocate ReSTIR neighbor offset buffer"))) {
            return;
        }
        if (restir_any_enabled &&
            (!allocate_svgf_buffer(device_restir_visibility_indices_, pixels * sizeof(int), "could not allocate ReSTIR visibility queue") ||
             !allocate_svgf_buffer(device_restir_visibility_count_, sizeof(int), "could not allocate ReSTIR visibility queue counter"))) {
            return;
        }
        if (restir_gi_enabled &&
            (!allocate_svgf_buffer(device_restir_gi_initial_samples_, pixels * sizeof(GpuRestirGiInitialSample), "could not allocate ReSTIR GI initial sample buffer") ||
             !allocate_svgf_buffer(device_restir_gi_secondary_compact_hits_, pixels * sizeof(GpuCompactHit), "could not allocate ReSTIR GI secondary compact hit buffer") ||
             !allocate_svgf_buffer(device_restir_gi_secondary_hits_, pixels * sizeof(GpuHit), "could not allocate ReSTIR GI secondary hit buffer") ||
             !allocate_svgf_buffer(device_restir_gi_secondary_results_, pixels * sizeof(int), "could not allocate ReSTIR GI secondary result buffer") ||
             !allocate_svgf_buffer(device_restir_gi_secondary_direct_, pixels * sizeof(GpuRestirReservoir), "could not allocate ReSTIR GI secondary direct reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_gi_initial_, pixels * sizeof(GpuPackedRestirGiReservoir), "could not allocate ReSTIR GI initial reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_gi_temporal_, pixels * sizeof(GpuPackedRestirGiReservoir), "could not allocate ReSTIR GI temporal reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_gi_history_, pixels * sizeof(GpuPackedRestirGiReservoir), "could not allocate ReSTIR GI history reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_gi_scratch_, pixels * sizeof(GpuPackedRestirGiReservoir), "could not allocate ReSTIR GI scratch reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_gi_current_surfaces_, pixels * sizeof(GpuRestirSurface), "could not allocate ReSTIR GI current surface buffer") ||
             !allocate_svgf_buffer(device_restir_gi_history_surfaces_, pixels * sizeof(GpuRestirSurface), "could not allocate ReSTIR GI history surface buffer") ||
             !allocate_svgf_buffer(device_restir_gi_temporal_states_, pixels * sizeof(GpuRestirGiTemporalState), "could not allocate ReSTIR GI temporal state buffer") ||
             !allocate_svgf_buffer(device_restir_gi_spatial_states_, pixels * sizeof(GpuRestirGiSpatialState), "could not allocate ReSTIR GI spatial state buffer") ||
             !allocate_svgf_buffer(device_restir_gi_visibility_rays_, pixels * sizeof(GpuRestirVisibilityRay), "could not allocate ReSTIR GI visibility ray buffer") ||
             !allocate_svgf_buffer(device_restir_gi_visibility_results_, pixels * sizeof(int), "could not allocate ReSTIR GI visibility result buffer"))) {
            return;
        }
        if (restir_pt_enabled &&
            (!allocate_svgf_buffer(device_restir_pt_path_states_, pixels * sizeof(GpuRestirPtPathState), "could not allocate ReSTIR PT path-state buffer") ||
             !allocate_svgf_buffer(device_restir_pt_compact_hits_, pixels * sizeof(GpuCompactHit), "could not allocate ReSTIR PT compact-hit buffer") ||
             !allocate_svgf_buffer(device_restir_pt_hits_, pixels * sizeof(GpuHit), "could not allocate ReSTIR PT hit buffer") ||
             !allocate_svgf_buffer(device_restir_pt_trace_results_, pixels * sizeof(int), "could not allocate ReSTIR PT trace-result buffer") ||
             !allocate_svgf_buffer(device_restir_pt_active_indices_, pixels * sizeof(int), "could not allocate ReSTIR PT active queue") ||
             !allocate_svgf_buffer(device_restir_pt_next_indices_, pixels * sizeof(int), "could not allocate ReSTIR PT next queue") ||
             !allocate_svgf_buffer(device_restir_pt_queue_counters_, sizeof(GpuRestirPtQueueCounters), "could not allocate ReSTIR PT queue counters") ||
             !allocate_svgf_buffer(device_restir_pt_nee_reservoirs_, pixels * sizeof(GpuRestirReservoir), "could not allocate ReSTIR PT NEE reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_pt_initial_, pixels * sizeof(GpuPackedRestirPtReservoir), "could not allocate ReSTIR PT initial reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_pt_temporal_, pixels * sizeof(GpuPackedRestirPtReservoir), "could not allocate ReSTIR PT temporal reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_pt_history_, pixels * sizeof(GpuPackedRestirPtReservoir), "could not allocate ReSTIR PT history reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_pt_scratch_, pixels * sizeof(GpuPackedRestirPtReservoir), "could not allocate ReSTIR PT scratch reservoir buffer") ||
             !allocate_svgf_buffer(device_restir_pt_current_surfaces_, pixels * sizeof(GpuRestirSurface), "could not allocate ReSTIR PT current-surface buffer") ||
             !allocate_svgf_buffer(device_restir_pt_history_surfaces_, pixels * sizeof(GpuRestirSurface), "could not allocate ReSTIR PT history-surface buffer") ||
             !allocate_svgf_buffer(device_restir_pt_temporal_states_, pixels * sizeof(GpuRestirPtTemporalState), "could not allocate ReSTIR PT temporal-state buffer") ||
              !allocate_svgf_buffer(device_restir_pt_spatial_states_, pixels * sizeof(GpuRestirPtSpatialState), "could not allocate ReSTIR PT spatial-state buffer") ||
              !allocate_svgf_buffer(device_restir_pt_visibility_rays_, pixels * sizeof(GpuRestirVisibilityRay), "could not allocate ReSTIR PT visibility-ray buffer") ||
              !allocate_svgf_buffer(device_restir_pt_visibility_results_, pixels * sizeof(int), "could not allocate ReSTIR PT visibility-result buffer") ||
              !allocate_svgf_buffer(device_restir_pt_sample_ids_, pixels * sizeof(uint32_t), "could not allocate ReSTIR PT sample-ID buffer") ||
              !allocate_svgf_buffer(device_restir_pt_duplication_counts_, pixels * sizeof(uint32_t), "could not allocate ReSTIR PT duplication-map buffer"))) {
            return;
        }
        if (upload_restir_neighbor_offsets) {
            static const std::vector<Vec2> neighbor_offsets = build_restir_neighbor_offsets();
            cuda_error = cudaMemcpy(device_restir_neighbor_offsets_, neighbor_offsets.data(),
                neighbor_offsets.size() * sizeof(Vec2), cudaMemcpyHostToDevice);
            if (cuda_error != cudaSuccess) {
                reset();
                render_cpu_fallback(scene, settings, framebuffer,
                    "could not upload ReSTIR neighbor offsets", cuda_error_text(cuda_error));
                return;
            }
        }
    }
    cached_pixels_ = pixels;

    Vec3* device_accumulation = static_cast<Vec3*>(device_accumulation_);
    uint32_t* device_rgba = static_cast<uint32_t*>(device_rgba_);
    GpuScene* device_scene = static_cast<GpuScene*>(device_scene_);

    if (has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform)) {
        cached_render_scene_valid_ = false;
    }
    if ((lightmap_rendering_enabled(settings) || irradiance_volume_rendering_enabled(settings)) && !cached_render_scene_valid_) {
        cached_render_scene_ = build_render_scene(scene);
        cached_render_scene_valid_ = true;
    }

    std::shared_ptr<Lightmap> lightmap;
    bool lightmap_rebuilt = false;
    const bool lightmap_enabled_changed = cached_lightmap_enabled_ != settings.use_lightmap;
    if (lightmap_rendering_enabled(settings)) {
        const bool lightmap_dirty = has_dirty(settings.dirty, RenderDirty::Lightmap) ||
            has_dirty(settings.dirty, RenderDirty::Geometry) ||
            has_dirty(settings.dirty, RenderDirty::Transform) ||
            has_dirty(settings.dirty, RenderDirty::Material) ||
            has_dirty(settings.dirty, RenderDirty::Texture) ||
            has_dirty(settings.dirty, RenderDirty::Environment);
        lightmap = update_lightmap(
            cached_lightmap_,
            cached_render_scene_,
            scene,
            settings,
            lightmap_dirty,
            lightmap_rebuilt);
    } else if (cached_lightmap_) {
        cached_lightmap_.reset();
        lightmap_rebuilt = true;
        apply_lightmap_to_render_scene(nullptr, cached_render_scene_);
        set_lightmap_progress_phase(settings, LightmapBakePhase::Idle);
    }

    std::shared_ptr<IrradianceVolume> irradiance_volume;
    bool irradiance_volume_rebuilt = false;
    const bool irradiance_volume_enabled_changed = cached_irradiance_volume_enabled_ != settings.use_irradiance_volume;
    if (irradiance_volume_rendering_enabled(settings)) {
        const bool volume_dirty = has_dirty(settings.dirty, RenderDirty::IrradianceVolume) ||
            has_dirty(settings.dirty, RenderDirty::Geometry) ||
            has_dirty(settings.dirty, RenderDirty::Material) ||
            has_dirty(settings.dirty, RenderDirty::Texture) ||
            has_dirty(settings.dirty, RenderDirty::Environment);
        irradiance_volume = update_irradiance_volume(
            cached_irradiance_volume_,
            cached_render_scene_,
            scene,
            settings,
            volume_dirty,
            irradiance_volume_rebuilt);
    } else if (cached_irradiance_volume_) {
        cached_irradiance_volume_.reset();
        irradiance_volume_rebuilt = true;
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Idle);
    }

    const bool full_upload = !scene_uploaded_ ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture) ||
        has_dirty(settings.dirty, RenderDirty::Environment) ||
        lightmap_rebuilt ||
        lightmap_enabled_changed;
    if (full_upload && !cached_render_scene_valid_) {
        cached_render_scene_ = build_render_scene(scene);
        cached_render_scene_valid_ = true;
    }
    const bool irradiance_volume_upload = full_upload || irradiance_volume_rebuilt || irradiance_volume_enabled_changed;
    const bool lightmap_upload = full_upload || lightmap_rebuilt || lightmap_enabled_changed;
    const auto upload_gpu_irradiance_volume = [&](const IrradianceVolume* volume, GpuIrradianceVolume& gpu_volume) {
        PackedGpuIrradianceVolume packed;
        if (!pack_irradiance_volume(volume, packed)) {
            return false;
        }
        if (!upload_buffer(device_irradiance_volume_directions_, cached_irradiance_volume_directions_, packed.directions) ||
            !upload_buffer(device_irradiance_volume_irradiance_, cached_irradiance_volume_irradiance_, packed.irradiance) ||
            !upload_buffer(device_irradiance_volume_grids_, cached_irradiance_volume_grids_, packed.grids) ||
            !upload_buffer(device_irradiance_volume_cells_, cached_irradiance_volume_cells_, packed.cell_subgrid_indices) ||
            !upload_buffer(device_irradiance_volume_debug_probes_, cached_irradiance_volume_debug_probes_, packed.debug_probes)) {
            return false;
        }
        gpu_volume = packed.volume;
        gpu_volume.directions = static_cast<Vec3*>(device_irradiance_volume_directions_);
        gpu_volume.irradiance = static_cast<Vec3*>(device_irradiance_volume_irradiance_);
        gpu_volume.grids = static_cast<GpuIrradianceVolumeGrid*>(device_irradiance_volume_grids_);
        gpu_volume.cell_subgrid_indices = static_cast<int*>(device_irradiance_volume_cells_);
        gpu_volume.debug_probes = static_cast<GpuIrradianceVolumeDebugProbe*>(device_irradiance_volume_debug_probes_);
        return true;
    };
    const auto upload_gpu_lightmap = [&](const Lightmap* source, GpuLightmap& gpu_lightmap) {
        PackedGpuLightmap packed;
        if (!pack_lightmap(source, packed)) {
            return false;
        }
        if (!upload_buffer(device_lightmap_texels_, cached_lightmap_texels_, packed.texels)) {
            return false;
        }
        gpu_lightmap = packed.lightmap;
        gpu_lightmap.texels = static_cast<Vec3*>(device_lightmap_texels_);
        return true;
    };
    const auto upload_restir_environment_sampler = [&](GpuEnvironmentSampler& gpu_sampler) {
        PackedGpuEnvironmentSampler packed;
        if (!build_environment_sampler(scene, packed)) {
            return false;
        }
        const int count = packed.sampler.texel_count;
        if (cached_restir_environment_texels_ != count) {
            cudaFree(device_restir_environment_pmf_);
            cudaFree(device_restir_environment_alias_probability_);
            cudaFree(device_restir_environment_alias_index_);
            device_restir_environment_pmf_ = nullptr;
            device_restir_environment_alias_probability_ = nullptr;
            device_restir_environment_alias_index_ = nullptr;
            cached_restir_environment_texels_ = 0;
        }
        if (count > 0) {
            if (!device_restir_environment_pmf_ && cudaMalloc(&device_restir_environment_pmf_, packed.pmf.size() * sizeof(float)) != cudaSuccess) return false;
            if (!device_restir_environment_alias_probability_ && cudaMalloc(&device_restir_environment_alias_probability_, packed.alias_probability.size() * sizeof(float)) != cudaSuccess) return false;
            if (!device_restir_environment_alias_index_ && cudaMalloc(&device_restir_environment_alias_index_, packed.alias_index.size() * sizeof(int)) != cudaSuccess) return false;
            if (cudaMemcpy(device_restir_environment_pmf_, packed.pmf.data(), packed.pmf.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(device_restir_environment_alias_probability_, packed.alias_probability.data(), packed.alias_probability.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(device_restir_environment_alias_index_, packed.alias_index.data(), packed.alias_index.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return false;
            cached_restir_environment_texels_ = count;
        }
        gpu_sampler = packed.sampler;
        gpu_sampler.pmf = static_cast<float*>(device_restir_environment_pmf_);
        gpu_sampler.alias_probability = static_cast<float*>(device_restir_environment_alias_probability_);
        gpu_sampler.alias_index = static_cast<int*>(device_restir_environment_alias_index_);
        return true;
    };
    const auto upload_restir_light_sampler = [&](GpuLightSampler& gpu_sampler) {
        PackedGpuLightSampler packed;
        if (!build_restir_light_sampler(scene, cached_render_scene_, packed)) {
            return false;
        }
        const int count = packed.sampler.light_count;
        if (cached_restir_light_count_ != count) {
            cudaFree(device_restir_light_pmf_);
            cudaFree(device_restir_light_alias_probability_);
            cudaFree(device_restir_light_alias_index_);
            device_restir_light_pmf_ = nullptr;
            device_restir_light_alias_probability_ = nullptr;
            device_restir_light_alias_index_ = nullptr;
            cached_restir_light_count_ = 0;
        }
        if (count > 0) {
            if (!device_restir_light_pmf_ && cudaMalloc(&device_restir_light_pmf_, packed.pmf.size() * sizeof(float)) != cudaSuccess) return false;
            if (!device_restir_light_alias_probability_ && cudaMalloc(&device_restir_light_alias_probability_, packed.alias_probability.size() * sizeof(float)) != cudaSuccess) return false;
            if (!device_restir_light_alias_index_ && cudaMalloc(&device_restir_light_alias_index_, packed.alias_index.size() * sizeof(int)) != cudaSuccess) return false;
            if (cudaMemcpy(device_restir_light_pmf_, packed.pmf.data(), packed.pmf.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(device_restir_light_alias_probability_, packed.alias_probability.data(), packed.alias_probability.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(device_restir_light_alias_index_, packed.alias_index.data(), packed.alias_index.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return false;
            cached_restir_light_count_ = count;
        }
        gpu_sampler = packed.sampler;
        gpu_sampler.pmf = static_cast<float*>(device_restir_light_pmf_);
        gpu_sampler.alias_probability = static_cast<float*>(device_restir_light_alias_probability_);
        gpu_sampler.alias_index = static_cast<int*>(device_restir_light_alias_index_);
        return true;
    };
    // The alias distribution depends on every environment edit, not just texel count.
    const bool restir_environment_upload = restir_any_enabled &&
        (full_upload || (!restir_was_enabled_ && !restir_gi_was_enabled_) || has_dirty(settings.dirty, RenderDirty::Environment) ||
            (device_restir_environment_pmf_ == nullptr && scene.environment.texture != nullptr));
    GpuEnvironmentSampler restir_environment_sampler;
    if (restir_environment_upload && !upload_restir_environment_sampler(restir_environment_sampler)) {
        reset();
        render_cpu_fallback(scene, settings, framebuffer, "could not upload ReSTIR environment sampler");
        return;
    }
    const bool restir_light_upload = restir_any_enabled &&
        (full_upload || (!restir_was_enabled_ && !restir_gi_was_enabled_) ||
            (device_restir_light_pmf_ == nullptr && !cached_render_scene_.light_triangle_indices.empty()));
    GpuLightSampler restir_light_sampler;
    if (restir_light_upload && !upload_restir_light_sampler(restir_light_sampler)) {
        reset();
        render_cpu_fallback(scene, settings, framebuffer, "could not upload ReSTIR local-light sampler");
        return;
    }

    if (full_upload) {
        PackedGpuScene packed;
        if (!pack_scene_from_render_scene(scene, settings, cached_render_scene_, packed)) {
            render_cpu_fallback(scene, settings, framebuffer, "could not pack scene for CUDA");
            return;
        }
        const bool upload_textures = texture_objects_.size() != packed.textures.size() || has_dirty(settings.dirty, RenderDirty::Texture);
        if (upload_textures ? !upload_texture_objects(scene, packed, texture_arrays_, texture_objects_) : !apply_cached_texture_objects(packed, texture_objects_)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, upload_textures ? "could not upload CUDA texture objects" : "could not apply cached CUDA texture objects");
            return;
        }
        packed.scene.environment_sampler = restir_environment_sampler;
        packed.scene.restir_light_sampler = restir_light_sampler;
        if (!upload_buffer(device_materials_, cached_materials_, packed.materials) ||
            !upload_buffer(device_textures_, cached_textures_, packed.textures) ||
            !upload_buffer(device_triangles_, cached_triangles_, packed.triangles) ||
            !upload_buffer(device_traversal_triangles_, cached_traversal_triangles_, packed.traversal_triangles) ||
            !upload_buffer(device_spheres_, cached_spheres_, packed.spheres) ||
            !upload_buffer(device_triangle_indices_, cached_triangle_indices_, packed.triangle_indices) ||
            !upload_buffer(device_light_indices_, cached_lights_, packed.light_indices) ||
            !upload_buffer(device_directional_lights_, cached_directional_lights_, packed.directional_lights) ||
            !upload_buffer(device_point_lights_, cached_point_lights_, packed.point_lights) ||
            !upload_buffer(device_bvh_nodes_, cached_bvh_nodes_, packed.bvh_nodes) ||
            !upload_buffer(device_traversal_bvh_nodes_, cached_traversal_bvh_nodes_, packed.traversal_bvh_nodes) ||
            !upload_buffer(device_traversal_bvh8_nodes_, cached_traversal_bvh8_nodes_, packed.traversal_bvh8_nodes) ||
            !upload_buffer(device_traversal_cwbvh_nodes_, cached_traversal_cwbvh_nodes_, packed.traversal_cwbvh_nodes) ||
            !upload_buffer(device_cwbvh_triangle_indices_, cached_cwbvh_triangle_indices_, packed.cwbvh_triangle_indices) ||
            !upload_buffer(device_cwbvh_triangles_, cached_cwbvh_triangles_, packed.cwbvh_triangles) ||
            !upload_buffer(device_mesh_instances_, cached_mesh_instances_, packed.mesh_instances) ||
            !upload_buffer(device_mesh_instance_indices_, cached_mesh_instance_indices_, packed.mesh_instance_indices) ||
            !upload_buffer(device_tlas_nodes_, cached_tlas_nodes_, packed.tlas_nodes) ||
            !upload_buffer(device_traversal_tlas_nodes_, cached_traversal_tlas_nodes_, packed.traversal_tlas_nodes)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA scene buffers");
            return;
        }
        packed.scene.materials = static_cast<GpuMaterial*>(device_materials_);
        packed.scene.textures = static_cast<GpuTexture*>(device_textures_);
        packed.scene.triangles = static_cast<GpuTriangle*>(device_triangles_);
        packed.scene.traversal_triangles = static_cast<GpuTraversalTriangle*>(device_traversal_triangles_);
        packed.scene.spheres = static_cast<GpuSphere*>(device_spheres_);
        packed.scene.triangle_indices = static_cast<int*>(device_triangle_indices_);
        packed.scene.light_indices = static_cast<int*>(device_light_indices_);
        packed.scene.directional_lights = static_cast<GpuDirectionalLight*>(device_directional_lights_);
        packed.scene.point_lights = static_cast<GpuPointLight*>(device_point_lights_);
        packed.scene.bvh_nodes = static_cast<GpuBvhNode*>(device_bvh_nodes_);
        packed.scene.traversal_bvh_nodes = static_cast<GpuTraversalBvhNode*>(device_traversal_bvh_nodes_);
        packed.scene.traversal_bvh8_nodes = static_cast<GpuTraversalBvh8Node*>(device_traversal_bvh8_nodes_);
        packed.scene.traversal_cwbvh_nodes = static_cast<GpuCwBvhNode*>(device_traversal_cwbvh_nodes_);
        packed.scene.cwbvh_triangle_indices = static_cast<int*>(device_cwbvh_triangle_indices_);
        packed.scene.cwbvh_triangles = static_cast<float4*>(device_cwbvh_triangles_);
        packed.scene.mesh_instances = static_cast<GpuMeshInstance*>(device_mesh_instances_);
        packed.scene.mesh_instance_indices = static_cast<int*>(device_mesh_instance_indices_);
        packed.scene.tlas_nodes = static_cast<GpuBvhNode*>(device_tlas_nodes_);
        packed.scene.traversal_tlas_nodes = static_cast<GpuTraversalBvhNode*>(device_traversal_tlas_nodes_);
        if (!upload_gpu_irradiance_volume(irradiance_volume.get(), packed.scene.irradiance_volume)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA irradiance volume");
            return;
        }
        if (!upload_gpu_lightmap(lightmap.get(), packed.scene.lightmap)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA lightmap");
            return;
        }
        cuda_error = cudaMemcpy(device_scene, &packed.scene, sizeof(GpuScene), cudaMemcpyHostToDevice);
        if (cuda_error != cudaSuccess) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA scene header", cuda_error_text(cuda_error));
            return;
        }
        scene_uploaded_ = true;
        cached_irradiance_volume_enabled_ = settings.use_irradiance_volume;
        cached_lightmap_enabled_ = settings.use_lightmap;
    } else {
        if (has_dirty(settings.dirty, RenderDirty::Camera) && !upload_camera(device_scene, scene.camera)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA camera");
            return;
        }
        if (has_dirty(settings.dirty, RenderDirty::Environment)) {
            GpuScene environment_scene;
            environment_scene.environment_sampler = restir_environment_sampler;
            if (!make_environment_gpu(scene, environment_scene) || !upload_environment(device_scene, environment_scene)) {
                reset();
                render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA environment");
                return;
            }
        }
        if (restir_environment_upload && !has_dirty(settings.dirty, RenderDirty::Environment) &&
            !upload_scene_field(device_scene, offsetof(GpuScene, environment_sampler), restir_environment_sampler)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not update ReSTIR environment sampler");
            return;
        }
        if (restir_light_upload && !upload_scene_field(device_scene, offsetof(GpuScene, restir_light_sampler), restir_light_sampler)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not update ReSTIR local-light sampler");
            return;
        }
        if (irradiance_volume_upload) {
            GpuIrradianceVolume gpu_volume;
            if (!upload_gpu_irradiance_volume(irradiance_volume.get(), gpu_volume) ||
                !upload_scene_field(device_scene, offsetof(GpuScene, irradiance_volume), gpu_volume)) {
                reset();
                render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA irradiance volume");
                return;
            }
            cached_irradiance_volume_enabled_ = settings.use_irradiance_volume;
        }
        if (lightmap_upload) {
            GpuLightmap gpu_lightmap;
            if (!upload_gpu_lightmap(lightmap.get(), gpu_lightmap) ||
                !upload_scene_field(device_scene, offsetof(GpuScene, lightmap), gpu_lightmap)) {
                reset();
                render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA lightmap");
                return;
            }
            cached_lightmap_enabled_ = settings.use_lightmap;
        }
    }

    const bool svgf_history_invalidated =
        settings.frame_index == 0u ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture) ||
        has_dirty(settings.dirty, RenderDirty::Environment) ||
        has_dirty(settings.dirty, RenderDirty::IrradianceVolume) ||
        has_dirty(settings.dirty, RenderDirty::Lightmap);
    if (svgf_history_invalidated) {
        svgf_history_valid_ = false;
    }

    const bool taa_history_invalidated =
        settings.frame_index == 0u ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture) ||
        has_dirty(settings.dirty, RenderDirty::Environment) ||
        has_dirty(settings.dirty, RenderDirty::IrradianceVolume) ||
        has_dirty(settings.dirty, RenderDirty::Lightmap);
    if (taa_history_invalidated) {
        taa_history_valid_ = false;
    }
    const bool restir_history_invalidated = !restir_enabled || !restir_was_enabled_ ||
        settings.cuda_restir_bias_correction != restir_history_bias_correction_ ||
        settings.cuda_restir_final_visibility_reuse != restir_history_visibility_reuse_ ||
        settings.frame_index == 0u ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture) ||
        has_dirty(settings.dirty, RenderDirty::Environment) ||
        has_dirty(settings.dirty, RenderDirty::Lightmap) ||
        has_dirty(settings.dirty, RenderDirty::IrradianceVolume);
    if (restir_history_invalidated) {
        restir_history_valid_ = false;
    }
    restir_was_enabled_ = restir_enabled;
    restir_history_bias_correction_ = settings.cuda_restir_bias_correction;
    restir_history_visibility_reuse_ = settings.cuda_restir_final_visibility_reuse;
    const bool restir_gi_history_invalidated = !restir_gi_enabled || !restir_gi_was_enabled_ ||
        settings.cuda_restir_gi_bias_correction != restir_gi_history_bias_correction_ ||
        settings.cuda_restir_gi_resampling != restir_gi_history_resampling_ ||
        settings.cuda_restir_gi_final_mis != restir_gi_history_final_mis_ ||
        settings.cuda_restir_gi_boiling_filter != restir_gi_history_boiling_filter_ ||
        fabsf(settings.cuda_restir_gi_secondary_roughness - restir_gi_history_secondary_roughness_) > 1e-6f ||
        settings.frame_index == 0u ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture) ||
        has_dirty(settings.dirty, RenderDirty::Environment) ||
        has_dirty(settings.dirty, RenderDirty::Lightmap) ||
        has_dirty(settings.dirty, RenderDirty::IrradianceVolume);
    if (restir_gi_history_invalidated) {
        restir_gi_history_valid_ = false;
    }
    restir_gi_was_enabled_ = restir_gi_enabled;
    restir_gi_history_bias_correction_ = settings.cuda_restir_gi_bias_correction;
    restir_gi_history_resampling_ = settings.cuda_restir_gi_resampling;
    restir_gi_history_final_mis_ = settings.cuda_restir_gi_final_mis;
    restir_gi_history_boiling_filter_ = settings.cuda_restir_gi_boiling_filter;
    restir_gi_history_secondary_roughness_ = settings.cuda_restir_gi_secondary_roughness;
    const int restir_pt_max_bounces = std::clamp(settings.cuda_restir_pt_max_bounces,
        2, std::max(2, std::min(settings.max_bounces, 8)));
    const bool restir_pt_history_invalidated = !restir_pt_enabled || !restir_pt_was_enabled_ ||
        settings.cuda_restir_pt_resampling != restir_pt_history_resampling_ ||
        settings.cuda_restir_pt_reconnection != restir_pt_history_reconnection_ ||
        restir_pt_max_bounces != restir_pt_history_max_bounces_ ||
        settings.frame_index == 0u ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture) ||
        has_dirty(settings.dirty, RenderDirty::Environment) ||
        has_dirty(settings.dirty, RenderDirty::Lightmap) ||
        has_dirty(settings.dirty, RenderDirty::IrradianceVolume);
    if (restir_pt_history_invalidated) {
        restir_pt_history_valid_ = false;
    }
    restir_pt_was_enabled_ = restir_pt_enabled;
    restir_pt_history_resampling_ = settings.cuda_restir_pt_resampling;
    restir_pt_history_reconnection_ = settings.cuda_restir_pt_reconnection;
    restir_pt_history_max_bounces_ = restir_pt_max_bounces;

    if (!svgf_denoising_enabled(settings) && (settings.frame_index == 0u || has_dirty(settings.dirty, RenderDirty::Render))) {
        cuda_error = cudaMemset(device_accumulation, 0, pixels * sizeof(Vec3));
        if (cuda_error != cudaSuccess) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not clear CUDA accumulation buffer", cuda_error_text(cuda_error));
            return;
        }
    }

    if (settings.frame_index == 0u) {
        std::fill(framebuffer.accumulation.begin(), framebuffer.accumulation.end(), Vec3{});
    }

    const dim3 block(16, 16);
    const dim3 grid((settings.width + block.x - 1) / block.x, (settings.height + block.y - 1) / block.y);
    const auto render_wavefront_samples = [&](
        Vec3* sample_sum,
        bool use_svgf_camera_jitter,
        GpuWavefrontSvgfAov svgf_aov) -> cudaError_t {
        const int pixel_count = static_cast<int>(pixels);
        const int queue_block_size = 256;
        const int intersect_block_size = 128;
        const auto queue_grid_size = [queue_block_size](int count) {
            return (count + queue_block_size - 1) / queue_block_size;
        };
        const auto intersect_grid_size = [intersect_block_size](int count) {
            return (count + intersect_block_size - 1) / intersect_block_size;
        };
        GpuWavefrontPaths paths{
            static_cast<Ray*>(device_wavefront_rays_),
            static_cast<Vec3*>(device_wavefront_throughputs_),
            static_cast<Vec3*>(device_wavefront_radiance_),
            static_cast<Vec3*>(device_wavefront_previous_positions_),
            static_cast<float*>(device_wavefront_previous_bsdf_pdfs_),
            static_cast<uint32_t*>(device_wavefront_rngs_),
            static_cast<int*>(device_wavefront_pixels_),
            static_cast<uint32_t*>(device_wavefront_states_),
        };
        GpuWavefrontIntersectPaths intersect_paths{
            static_cast<Ray*>(device_wavefront_rays_),
            static_cast<Vec3*>(device_wavefront_throughputs_),
            static_cast<Vec3*>(device_wavefront_radiance_),
            static_cast<uint32_t*>(device_wavefront_rngs_),
            static_cast<int*>(device_wavefront_pixels_),
            static_cast<uint32_t*>(device_wavefront_states_),
        };
        GpuCompactHit* compact_hits = static_cast<GpuCompactHit*>(device_wavefront_compact_hits_);
        GpuHit* hits = static_cast<GpuHit*>(device_wavefront_hits_);
        int* active_indices = static_cast<int*>(device_wavefront_active_indices_);
        int* next_indices = static_cast<int*>(device_wavefront_next_indices_);
        int* shade_indices = static_cast<int*>(device_wavefront_shade_indices_);
        int* shadow_indices = static_cast<int*>(device_wavefront_shadow_indices_);
        int* gi_indices = static_cast<int*>(device_wavefront_gi_indices_);
        int* bsdf_indices = static_cast<int*>(device_wavefront_bsdf_indices_);
        GpuWavefrontQueueCounters* queue_counters =
            static_cast<GpuWavefrontQueueCounters*>(device_wavefront_queue_counters_);
        Vec3* samples = static_cast<Vec3*>(device_wavefront_samples_);
        GpuRestirReservoir* restir_initial = static_cast<GpuRestirReservoir*>(device_restir_initial_);
        GpuRestirReservoir* restir_temporal = static_cast<GpuRestirReservoir*>(device_restir_temporal_);
        GpuRestirReservoir* restir_history = static_cast<GpuRestirReservoir*>(device_restir_history_);
        GpuRestirReservoir* restir_scratch = static_cast<GpuRestirReservoir*>(device_restir_scratch_);
        GpuRestirSurface* restir_history_surfaces = static_cast<GpuRestirSurface*>(device_restir_history_surfaces_);
        GpuRestirSurface* restir_current_surfaces = static_cast<GpuRestirSurface*>(device_restir_current_surfaces_);
        GpuRestirTemporalState* restir_temporal_states =
            static_cast<GpuRestirTemporalState*>(device_restir_temporal_states_);
        GpuRestirSpatialState* restir_spatial_states =
            static_cast<GpuRestirSpatialState*>(device_restir_spatial_states_);
        GpuRestirVisibilityRay* restir_visibility_rays =
            static_cast<GpuRestirVisibilityRay*>(device_restir_visibility_rays_);
        int* restir_visibility_results = static_cast<int*>(device_restir_visibility_results_);
        int* restir_visibility_indices = static_cast<int*>(device_restir_visibility_indices_);
        int* restir_visibility_count = static_cast<int*>(device_restir_visibility_count_);
        const Vec2* restir_neighbor_offsets = static_cast<const Vec2*>(device_restir_neighbor_offsets_);
        GpuRestirGiInitialSample* restir_gi_initial_samples =
            static_cast<GpuRestirGiInitialSample*>(device_restir_gi_initial_samples_);
        GpuCompactHit* restir_gi_secondary_compact_hits =
            static_cast<GpuCompactHit*>(device_restir_gi_secondary_compact_hits_);
        GpuHit* restir_gi_secondary_hits = static_cast<GpuHit*>(device_restir_gi_secondary_hits_);
        int* restir_gi_secondary_results = static_cast<int*>(device_restir_gi_secondary_results_);
        GpuRestirReservoir* restir_gi_secondary_direct =
            static_cast<GpuRestirReservoir*>(device_restir_gi_secondary_direct_);
        GpuPackedRestirGiReservoir* restir_gi_initial =
            static_cast<GpuPackedRestirGiReservoir*>(device_restir_gi_initial_);
        GpuPackedRestirGiReservoir* restir_gi_temporal =
            static_cast<GpuPackedRestirGiReservoir*>(device_restir_gi_temporal_);
        GpuPackedRestirGiReservoir* restir_gi_history =
            static_cast<GpuPackedRestirGiReservoir*>(device_restir_gi_history_);
        GpuPackedRestirGiReservoir* restir_gi_scratch =
            static_cast<GpuPackedRestirGiReservoir*>(device_restir_gi_scratch_);
        GpuRestirSurface* restir_gi_current_surfaces =
            static_cast<GpuRestirSurface*>(device_restir_gi_current_surfaces_);
        GpuRestirSurface* restir_gi_history_surfaces =
            static_cast<GpuRestirSurface*>(device_restir_gi_history_surfaces_);
        GpuRestirGiTemporalState* restir_gi_temporal_states =
            static_cast<GpuRestirGiTemporalState*>(device_restir_gi_temporal_states_);
        GpuRestirGiSpatialState* restir_gi_spatial_states =
            static_cast<GpuRestirGiSpatialState*>(device_restir_gi_spatial_states_);
        GpuRestirVisibilityRay* restir_gi_visibility_rays =
            static_cast<GpuRestirVisibilityRay*>(device_restir_gi_visibility_rays_);
        int* restir_gi_visibility_results = static_cast<int*>(device_restir_gi_visibility_results_);
        GpuRestirPtPathState* restir_pt_path_states =
            static_cast<GpuRestirPtPathState*>(device_restir_pt_path_states_);
        GpuCompactHit* restir_pt_compact_hits =
            static_cast<GpuCompactHit*>(device_restir_pt_compact_hits_);
        GpuHit* restir_pt_hits = static_cast<GpuHit*>(device_restir_pt_hits_);
        int* restir_pt_trace_results = static_cast<int*>(device_restir_pt_trace_results_);
        int* restir_pt_active_indices = static_cast<int*>(device_restir_pt_active_indices_);
        int* restir_pt_next_indices = static_cast<int*>(device_restir_pt_next_indices_);
        GpuRestirPtQueueCounters* restir_pt_queue_counters =
            static_cast<GpuRestirPtQueueCounters*>(device_restir_pt_queue_counters_);
        GpuRestirReservoir* restir_pt_nee_reservoirs =
            static_cast<GpuRestirReservoir*>(device_restir_pt_nee_reservoirs_);
        GpuPackedRestirPtReservoir* restir_pt_initial =
            static_cast<GpuPackedRestirPtReservoir*>(device_restir_pt_initial_);
        GpuPackedRestirPtReservoir* restir_pt_temporal =
            static_cast<GpuPackedRestirPtReservoir*>(device_restir_pt_temporal_);
        GpuPackedRestirPtReservoir* restir_pt_history =
            static_cast<GpuPackedRestirPtReservoir*>(device_restir_pt_history_);
        GpuPackedRestirPtReservoir* restir_pt_scratch =
            static_cast<GpuPackedRestirPtReservoir*>(device_restir_pt_scratch_);
        GpuRestirSurface* restir_pt_current_surfaces =
            static_cast<GpuRestirSurface*>(device_restir_pt_current_surfaces_);
        GpuRestirSurface* restir_pt_history_surfaces =
            static_cast<GpuRestirSurface*>(device_restir_pt_history_surfaces_);
        GpuRestirPtTemporalState* restir_pt_temporal_states =
            static_cast<GpuRestirPtTemporalState*>(device_restir_pt_temporal_states_);
        GpuRestirPtSpatialState* restir_pt_spatial_states =
            static_cast<GpuRestirPtSpatialState*>(device_restir_pt_spatial_states_);
        GpuRestirVisibilityRay* restir_pt_visibility_rays =
            static_cast<GpuRestirVisibilityRay*>(device_restir_pt_visibility_rays_);
        int* restir_pt_visibility_results = static_cast<int*>(device_restir_pt_visibility_results_);
        uint32_t* restir_pt_sample_ids = static_cast<uint32_t*>(device_restir_pt_sample_ids_);
        uint32_t* restir_pt_duplication_counts =
            static_cast<uint32_t*>(device_restir_pt_duplication_counts_);
        cudaError_t render_error = cudaMemset(sample_sum, 0, pixels * sizeof(Vec3));
        bool has_alpha_visibility = false;
        bool has_transmission = false;
        for (const std::shared_ptr<Material>& material : scene.materials) {
            if (!material) {
                continue;
            }
            has_alpha_visibility = has_alpha_visibility || material->alpha_mode != AlphaMode::Opaque;
            const BrdfModel model = material->model();
            has_transmission = has_transmission ||
                model == BrdfModel::Dielectric ||
                model == BrdfModel::DiffuseTransmission;
            if (const auto* standard = dynamic_cast<const StandardSurfaceMaterial*>(material.get())) {
                has_transmission = has_transmission ||
                    standard->transmission_weight > 0.0f ||
                    standard->transmission_input.texture != nullptr;
            }
        }
        constexpr int kExtraTransmissionBounces = 12;
        const int max_path_steps = std::max(1, settings.max_bounces) +
            (has_transmission ? kExtraTransmissionBounces : 0);
        const bool has_direct_lights =
            !cached_render_scene_.light_triangle_indices.empty() ||
            !scene.directional_lights.empty() ||
            !scene.point_lights.empty() || restir_any_enabled;
        const bool has_restir_local_lights =
            !cached_render_scene_.light_triangle_indices.empty() ||
            !scene.directional_lights.empty() ||
            !scene.point_lights.empty();
        const bool has_restir_environment = scene.environment.strength > 0.0f &&
            (scene.environment.texture != nullptr || scene.environment.color.x > 0.0f ||
                scene.environment.color.y > 0.0f || scene.environment.color.z > 0.0f);
        const bool use_two_level = use_two_level_accel(cached_render_scene_, settings.acceleration_structure);
        const bool use_wide_bvh = use_wavefront_bvh8_layout(cached_render_scene_, settings);
        const bool use_cwbvh = use_wavefront_cwbvh_layout(cached_render_scene_, settings);

        for (int sample_index = 0;
             render_error == cudaSuccess && sample_index < settings.samples_per_pixel;
             ++sample_index) {
            const uint32_t restir_gi_sequence_index = settings.frame_index *
                static_cast<uint32_t>(std::max(1, settings.samples_per_pixel)) +
                static_cast<uint32_t>(sample_index);
            int* sample_active_indices = active_indices;
            int* sample_next_indices = next_indices;
            const int full_queue_grid = queue_grid_size(pixel_count);
            const int full_intersect_grid = intersect_grid_size(pixel_count);
            if (restir_enabled) {
                restir_clear_surfaces_kernel<<<full_queue_grid, queue_block_size>>>(restir_current_surfaces, pixel_count);
            }
            if (restir_gi_enabled) {
                restir_clear_surfaces_kernel<<<full_queue_grid, queue_block_size>>>(restir_gi_current_surfaces, pixel_count);
                cudaMemset(restir_gi_initial_samples, 0, pixels * sizeof(GpuRestirGiInitialSample));
                cudaMemset(restir_gi_initial, 0, pixels * sizeof(GpuPackedRestirGiReservoir));
            }
            if (restir_pt_enabled) {
                restir_clear_surfaces_kernel<<<full_queue_grid, queue_block_size>>>(restir_pt_current_surfaces, pixel_count);
                cudaMemset(restir_pt_path_states, 0, pixels * sizeof(GpuRestirPtPathState));
                cudaMemset(restir_pt_initial, 0, pixels * sizeof(GpuPackedRestirPtReservoir));
            }
            {
                const ScopedNvtxRange range("wavefront initialize");
                wavefront_initialize_kernel<<<queue_grid_size(pixel_count), queue_block_size>>>(
                    device_scene,
                    settings,
                    paths,
                    sample_active_indices,
                    queue_counters,
                    samples,
                    svgf_aov,
                    pixel_count,
                    sample_index,
                    use_svgf_camera_jitter);
            }
            for (int step = 0; render_error == cudaSuccess && step < max_path_steps; ++step) {
                const ScopedNvtxRange step_range("wavefront step");
                GpuRestirReservoir* restir_final = nullptr;
                wavefront_prepare_step_kernel<<<1, 1>>>(queue_counters);

                {
                    const ScopedNvtxRange range("wavefront intersect");
                    #define LT_LAUNCH_WAVEFRONT_INTERSECT(ALPHA, TWO_LEVEL, PRIMARY, LAYOUT) \
                        wavefront_intersect_kernel<ALPHA, TWO_LEVEL, PRIMARY, LAYOUT><<<full_intersect_grid, intersect_block_size>>>( \
                            device_scene, settings, intersect_paths, compact_hits, sample_active_indices, \
                            shade_indices, queue_counters, samples)

                    if (has_alpha_visibility && use_two_level && use_cwbvh && step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, true, true, GpuTraversalLayout::CwBvh);
                    } else if (has_alpha_visibility && use_two_level && use_cwbvh) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, true, false, GpuTraversalLayout::CwBvh);
                    } else if (has_alpha_visibility && use_two_level && use_wide_bvh && step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, true, true, GpuTraversalLayout::Bvh8);
                    } else if (has_alpha_visibility && use_two_level && use_wide_bvh) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, true, false, GpuTraversalLayout::Bvh8);
                    } else if (has_alpha_visibility && use_two_level && step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, true, true, GpuTraversalLayout::Binary);
                    } else if (has_alpha_visibility && use_two_level) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, true, false, GpuTraversalLayout::Binary);
                    } else if (has_alpha_visibility && step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, false, true, GpuTraversalLayout::Binary);
                    } else if (has_alpha_visibility) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(true, false, false, GpuTraversalLayout::Binary);
                    } else if (use_two_level && use_cwbvh && step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, true, true, GpuTraversalLayout::CwBvh);
                    } else if (use_two_level && use_cwbvh) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, true, false, GpuTraversalLayout::CwBvh);
                    } else if (use_two_level && use_wide_bvh && step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, true, true, GpuTraversalLayout::Bvh8);
                    } else if (use_two_level && use_wide_bvh) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, true, false, GpuTraversalLayout::Bvh8);
                    } else if (use_two_level && step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, true, true, GpuTraversalLayout::Binary);
                    } else if (use_two_level) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, true, false, GpuTraversalLayout::Binary);
                    } else if (step == 0) {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, false, true, GpuTraversalLayout::Binary);
                    } else {
                        LT_LAUNCH_WAVEFRONT_INTERSECT(false, false, false, GpuTraversalLayout::Binary);
                    }
                    #undef LT_LAUNCH_WAVEFRONT_INTERSECT
                }

                {
                    const ScopedNvtxRange range("wavefront surface setup");
                    wavefront_direct_light_kernel<<<full_queue_grid, queue_block_size>>>(
                        device_scene, settings, paths, compact_hits, hits, shade_indices,
                        gi_indices, shadow_indices, bsdf_indices, queue_counters, svgf_aov, samples);
                }

                if (restir_enabled && step == 0) {
                    {
                        const ScopedNvtxRange range("ReSTIR initial candidates");
                        #define LT_LAUNCH_RESTIR_INITIAL(LOCAL, ENVIRONMENT) \
                            restir_initial_candidates_kernel<LOCAL, ENVIRONMENT><<<full_queue_grid, queue_block_size>>>( \
                                device_scene, settings, paths, hits, shadow_indices, queue_counters, \
                                restir_initial, restir_current_surfaces)
                        if (has_restir_local_lights && has_restir_environment) {
                            LT_LAUNCH_RESTIR_INITIAL(true, true);
                        } else if (has_restir_local_lights) {
                            LT_LAUNCH_RESTIR_INITIAL(true, false);
                        } else if (has_restir_environment) {
                            LT_LAUNCH_RESTIR_INITIAL(false, true);
                        } else {
                            LT_LAUNCH_RESTIR_INITIAL(false, false);
                        }
                        #undef LT_LAUNCH_RESTIR_INITIAL
                    }
                    {
                        const ScopedNvtxRange range("ReSTIR initial visibility rays");
                        cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                        restir_generate_visibility_rays_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, shadow_indices, queue_counters,
                            restir_initial, restir_visibility_rays, restir_visibility_results,
                            restir_visibility_indices, restir_visibility_count);
                    }
                    {
                        const ScopedNvtxRange range("ReSTIR initial visibility trace");
                        if (has_alpha_visibility || has_transmission) {
                            launch_restir_visibility_trace<true>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_visibility_rays, restir_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        } else {
                            launch_restir_visibility_trace<false>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_visibility_rays, restir_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        }
                    }
                    {
                        const ScopedNvtxRange range("ReSTIR initial visibility resolve");
                        restir_resolve_initial_visibility_kernel<<<full_queue_grid, queue_block_size>>>(
                            paths, shadow_indices, queue_counters, restir_initial, restir_visibility_results);
                    }
                    {
                        const ScopedNvtxRange range("ReSTIR temporal reuse");
                        restir_temporal_resample_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, shadow_indices, queue_counters,
                            restir_initial, restir_history, restir_current_surfaces, restir_history_surfaces,
                            restir_temporal, restir_temporal_states, restir_history_camera_, restir_history_valid_);
                        restir_temporal_finalize_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, shadow_indices, queue_counters, restir_initial,
                            restir_history_surfaces, restir_temporal_states, restir_temporal);
                        const dim3 boiling_block(16, 16);
                        const dim3 boiling_grid(
                            (settings.width + boiling_block.x - 1) / boiling_block.x,
                            (settings.height + boiling_block.y - 1) / boiling_block.y);
                        restir_boiling_filter_kernel<<<boiling_grid, boiling_block>>>(
                            restir_current_surfaces, restir_temporal, settings.width, settings.height);
                    }
                    restir_final = restir_temporal;
                    if constexpr (kRestirSpatialSamples > 0) {
                        const ScopedNvtxRange range("ReSTIR spatial reuse");
                        restir_spatial_resample_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, shadow_indices, queue_counters,
                            restir_temporal, restir_current_surfaces, restir_neighbor_offsets,
                            restir_scratch, restir_spatial_states);
                        restir_spatial_finalize_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, shadow_indices, queue_counters,
                            restir_temporal, restir_current_surfaces, restir_neighbor_offsets,
                            restir_spatial_states, restir_scratch);
                        restir_final = restir_scratch;
                    }
                    {
                        const ScopedNvtxRange range("ReSTIR visibility rays");
                        cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                        restir_generate_visibility_rays_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, shadow_indices, queue_counters,
                            restir_final, restir_visibility_rays, restir_visibility_results,
                            restir_visibility_indices, restir_visibility_count);
                    }
                    {
                        const ScopedNvtxRange range("ReSTIR visibility trace");
                        if (has_alpha_visibility || has_transmission) {
                            launch_restir_visibility_trace<true>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_visibility_rays, restir_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        } else {
                            launch_restir_visibility_trace<false>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_visibility_rays, restir_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        }
                    }
                    {
                        const ScopedNvtxRange range("ReSTIR visibility resolve");
                        restir_resolve_visibility_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, shadow_indices, queue_counters,
                            restir_final, restir_visibility_results);
                    }
                }

                if (settings.use_lightmap || settings.use_irradiance_volume) {
                    wavefront_gi_kernel<<<full_queue_grid, queue_block_size>>>(
                        device_scene, settings, paths, hits, gi_indices,
                        bsdf_indices, queue_counters, samples);
                }

                if (settings.sampling_mode != PathSamplingMode::Unidirectional) {
                    if (has_direct_lights && !(restir_enabled && step == 0)) {
                        const ScopedNvtxRange range("wavefront direct visibility");
                        wavefront_direct_visibility_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, shadow_indices,
                            queue_counters, samples);
                    }
                    if (restir_enabled && step == 0) {
                        if constexpr (kRestirSpatialSamples > 0) {
                            std::swap(device_restir_history_, device_restir_scratch_);
                        } else {
                            std::swap(device_restir_history_, device_restir_temporal_);
                        }
                        std::swap(device_restir_history_surfaces_, device_restir_current_surfaces_);
                        restir_history = static_cast<GpuRestirReservoir*>(device_restir_history_);
                        restir_scratch = static_cast<GpuRestirReservoir*>(device_restir_scratch_);
                        restir_temporal = static_cast<GpuRestirReservoir*>(device_restir_temporal_);
                        restir_history_surfaces = static_cast<GpuRestirSurface*>(device_restir_history_surfaces_);
                        restir_current_surfaces = static_cast<GpuRestirSurface*>(device_restir_current_surfaces_);
                        restir_history_camera_ = scene.camera;
                        restir_history_valid_ = true;
                    }
                    if (restir_pt_enabled && step == 0) {
                        const ScopedNvtxRange pt_range("ReSTIR PT");
                        cudaMemsetAsync(restir_pt_path_states, 0,
                            pixels * sizeof(GpuRestirPtPathState));
                        cudaMemsetAsync(restir_pt_initial, 0,
                            pixels * sizeof(GpuPackedRestirPtReservoir));
                        cudaMemsetAsync(restir_pt_current_surfaces, 0,
                            pixels * sizeof(GpuRestirSurface));
                        restir_pt_reset_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                        restir_pt_setup_initial_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, bsdf_indices, queue_counters,
                            restir_pt_path_states, restir_pt_initial, restir_pt_current_surfaces,
                            restir_pt_active_indices, restir_pt_queue_counters, samples,
                            restir_gi_sequence_index);

                        int* pt_active = restir_pt_active_indices;
                        int* pt_next = restir_pt_next_indices;
                        const int pt_trace_steps = std::clamp(settings.cuda_restir_pt_max_bounces,
                            2, std::max(2, std::min(settings.max_bounces, 8))) + kRestirPtExtraDeltaBudget;
                        for (int pt_step = 0; pt_step < pt_trace_steps; ++pt_step) {
                            if (has_alpha_visibility) {
                                launch_restir_pt_trace<true>(device_scene, restir_pt_path_states,
                                    restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                    restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            } else {
                                launch_restir_pt_trace<false>(device_scene, restir_pt_path_states,
                                    restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                    restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            }
                            restir_pt_resolve_trace_kernel<<<full_queue_grid, queue_block_size>>>(
                                device_scene, settings, restir_pt_path_states, restir_pt_compact_hits,
                                restir_pt_hits, restir_pt_trace_results, pt_active, restir_pt_queue_counters);

                            cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                            #define LT_LAUNCH_RESTIR_PT_NEE(LOCAL, ENVIRONMENT) \
                                restir_pt_generate_nee_kernel<LOCAL, ENVIRONMENT> \
                                    <<<full_queue_grid, queue_block_size>>>( \
                                        device_scene, settings, restir_pt_path_states, restir_pt_hits, \
                                        restir_pt_trace_results, pt_active, restir_pt_queue_counters, \
                                        restir_pt_nee_reservoirs, restir_pt_visibility_rays, \
                                        restir_pt_visibility_results, restir_visibility_indices, \
                                        restir_visibility_count)
                            if (has_restir_local_lights && has_restir_environment) {
                                LT_LAUNCH_RESTIR_PT_NEE(true, true);
                            } else if (has_restir_local_lights) {
                                LT_LAUNCH_RESTIR_PT_NEE(true, false);
                            } else if (has_restir_environment) {
                                LT_LAUNCH_RESTIR_PT_NEE(false, true);
                            } else {
                                LT_LAUNCH_RESTIR_PT_NEE(false, false);
                            }
                            #undef LT_LAUNCH_RESTIR_PT_NEE
                            if (has_alpha_visibility || has_transmission) {
                                launch_restir_visibility_trace<true>(device_scene, paths,
                                    restir_visibility_indices, restir_visibility_count,
                                    restir_pt_visibility_rays, restir_pt_visibility_results,
                                    use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            } else {
                                launch_restir_visibility_trace<false>(device_scene, paths,
                                    restir_visibility_indices, restir_visibility_count,
                                    restir_pt_visibility_rays, restir_pt_visibility_results,
                                    use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            }
                            restir_pt_stream_nee_continue_kernel<<<full_queue_grid, queue_block_size>>>(
                                device_scene, settings, restir_pt_path_states, restir_pt_hits,
                                restir_pt_trace_results, restir_pt_nee_reservoirs,
                                restir_pt_visibility_results, pt_active, pt_next,
                                restir_pt_queue_counters);
                            restir_pt_promote_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                            std::swap(pt_active, pt_next);
                        }
                        restir_pt_finalize_initial_kernel<<<full_queue_grid, queue_block_size>>>(
                            restir_pt_path_states, restir_pt_initial, pixel_count);

                        GpuPackedRestirPtReservoir* pt_final = restir_pt_initial;
                        if (settings.cuda_restir_pt_resampling != RestirPtResamplingMode::None) {
                            cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                            restir_pt_temporal_prepare_kernel<<<full_queue_grid, queue_block_size>>>(
                                device_scene, settings, restir_pt_current_surfaces,
                                restir_pt_history_surfaces, restir_pt_history,
                                settings.cuda_restir_pt_resampling == RestirPtResamplingMode::TemporalSpatial
                                    ? restir_pt_duplication_counts : nullptr,
                                restir_pt_temporal_states,
                                restir_pt_visibility_rays,
                                restir_pt_visibility_results, restir_visibility_indices,
                                restir_visibility_count, restir_pt_path_states,
                                restir_pt_history_camera_, restir_pt_history_valid_,
                                restir_gi_sequence_index, pixel_count);
                            restir_pt_reset_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                            pt_active = restir_pt_active_indices;
                            pt_next = restir_pt_next_indices;
                            restir_pt_replay_setup_kernel<GpuRestirPtTemporalState>
                                <<<full_queue_grid, queue_block_size>>>(
                                    device_scene, restir_pt_path_states, restir_pt_temporal_states,
                                    restir_pt_history, restir_pt_current_surfaces,
                                    pt_active, restir_pt_queue_counters, pixel_count);
                            for (int replay_step = 0; replay_step < 8; ++replay_step) {
                                if (has_alpha_visibility) {
                                    launch_restir_pt_trace<true>(device_scene, restir_pt_path_states,
                                        restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                        restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                } else {
                                    launch_restir_pt_trace<false>(device_scene, restir_pt_path_states,
                                        restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                        restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                }
                                restir_pt_replay_resolve_kernel<GpuRestirPtTemporalState>
                                    <<<full_queue_grid, queue_block_size>>>(
                                        device_scene, settings, restir_pt_path_states, restir_pt_temporal_states,
                                        restir_pt_compact_hits, restir_pt_trace_results, pt_active, pt_next,
                                        restir_pt_queue_counters, restir_pt_scratch,
                                        restir_pt_visibility_rays, restir_pt_visibility_results,
                                        restir_visibility_indices, restir_visibility_count);
                                restir_pt_promote_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                                std::swap(pt_active, pt_next);
                            }
                            if (has_alpha_visibility || has_transmission) {
                                launch_restir_visibility_trace<true>(device_scene, paths,
                                    restir_visibility_indices, restir_visibility_count,
                                    restir_pt_visibility_rays, restir_pt_visibility_results,
                                    use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            } else {
                                launch_restir_visibility_trace<false>(device_scene, paths,
                                    restir_visibility_indices, restir_visibility_count,
                                    restir_pt_visibility_rays, restir_pt_visibility_results,
                                    use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            }
                            restir_pt_temporal_select_kernel<<<full_queue_grid, queue_block_size>>>(
                                device_scene, restir_pt_current_surfaces, restir_pt_history_surfaces,
                                restir_pt_initial, restir_pt_history, restir_pt_scratch,
                                restir_pt_temporal_states,
                                restir_pt_visibility_results,
                                restir_pt_temporal, restir_gi_sequence_index, pixel_count);
                            cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                            restir_pt_inverse_prepare_kernel<GpuRestirPtTemporalState>
                                <<<full_queue_grid, queue_block_size>>>(
                                    device_scene, settings, restir_pt_history_surfaces, restir_pt_temporal,
                                    restir_pt_temporal_states, restir_pt_path_states,
                                    restir_pt_visibility_rays, restir_pt_visibility_results,
                                    restir_visibility_indices, restir_visibility_count, pixel_count);
                            restir_pt_reset_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                            pt_active = restir_pt_active_indices;
                            pt_next = restir_pt_next_indices;
                            restir_pt_inverse_replay_setup_kernel<GpuRestirPtTemporalState>
                                <<<full_queue_grid, queue_block_size>>>(
                                    device_scene, restir_pt_path_states, restir_pt_temporal_states,
                                    restir_pt_temporal, restir_pt_history_surfaces,
                                    pt_active, restir_pt_queue_counters, pixel_count);
                            for (int replay_step = 0; replay_step < 8; ++replay_step) {
                                if (has_alpha_visibility) {
                                    launch_restir_pt_trace<true>(device_scene, restir_pt_path_states,
                                        restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                        restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                } else {
                                    launch_restir_pt_trace<false>(device_scene, restir_pt_path_states,
                                        restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                        restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                }
                                restir_pt_replay_resolve_kernel<GpuRestirPtTemporalState>
                                    <<<full_queue_grid, queue_block_size>>>(
                                        device_scene, settings, restir_pt_path_states, restir_pt_temporal_states,
                                        restir_pt_compact_hits, restir_pt_trace_results, pt_active, pt_next,
                                        restir_pt_queue_counters, restir_pt_scratch,
                                        restir_pt_visibility_rays, restir_pt_visibility_results,
                                        restir_visibility_indices, restir_visibility_count);
                                restir_pt_promote_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                                std::swap(pt_active, pt_next);
                            }
                            if (has_alpha_visibility || has_transmission) {
                                launch_restir_visibility_trace<true>(device_scene, paths,
                                    restir_visibility_indices, restir_visibility_count,
                                    restir_pt_visibility_rays, restir_pt_visibility_results,
                                    use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            } else {
                                launch_restir_visibility_trace<false>(device_scene, paths,
                                    restir_visibility_indices, restir_visibility_count,
                                    restir_pt_visibility_rays, restir_pt_visibility_results,
                                    use_two_level, use_wide_bvh, use_cwbvh,
                                    full_intersect_grid, intersect_block_size);
                            }
                            restir_pt_pairwise_normalize_kernel<GpuRestirPtTemporalState>
                                <<<full_queue_grid, queue_block_size>>>(
                                    device_scene, restir_pt_history_surfaces, restir_pt_initial,
                                    restir_pt_temporal, restir_pt_scratch, restir_pt_temporal_states,
                                    restir_pt_visibility_results, restir_pt_temporal, pixel_count);
                            const dim3 boiling_block(16, 16);
                            const dim3 boiling_grid(
                                (settings.width + boiling_block.x - 1) / boiling_block.x,
                                (settings.height + boiling_block.y - 1) / boiling_block.y);
                            restir_pt_boiling_filter_kernel<<<boiling_grid, boiling_block>>>(
                                restir_pt_temporal, restir_pt_scratch, settings.width, settings.height);
                            pt_final = restir_pt_scratch;
                            if (settings.cuda_restir_pt_resampling == RestirPtResamplingMode::TemporalSpatial) {
                                cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                                restir_pt_spatial_prepare_kernel<<<full_queue_grid, queue_block_size>>>(
                                    device_scene, settings, restir_pt_current_surfaces, pt_final,
                                    restir_pt_spatial_states, restir_pt_visibility_rays,
                                    restir_pt_visibility_results, restir_visibility_indices,
                                    restir_visibility_count, restir_pt_path_states,
                                    restir_gi_sequence_index, pixel_count);
                                restir_pt_reset_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                                pt_active = restir_pt_active_indices;
                                pt_next = restir_pt_next_indices;
                                restir_pt_replay_setup_kernel<GpuRestirPtSpatialState>
                                    <<<full_queue_grid, queue_block_size>>>(
                                        device_scene, restir_pt_path_states, restir_pt_spatial_states,
                                        pt_final, restir_pt_current_surfaces,
                                        pt_active, restir_pt_queue_counters, pixel_count);
                                for (int replay_step = 0; replay_step < 8; ++replay_step) {
                                    if (has_alpha_visibility) {
                                        launch_restir_pt_trace<true>(device_scene, restir_pt_path_states,
                                            restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                            restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                            full_intersect_grid, intersect_block_size);
                                    } else {
                                        launch_restir_pt_trace<false>(device_scene, restir_pt_path_states,
                                            restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                            restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                            full_intersect_grid, intersect_block_size);
                                    }
                                    restir_pt_replay_resolve_kernel<GpuRestirPtSpatialState>
                                        <<<full_queue_grid, queue_block_size>>>(
                                            device_scene, settings, restir_pt_path_states, restir_pt_spatial_states,
                                            restir_pt_compact_hits, restir_pt_trace_results, pt_active, pt_next,
                                            restir_pt_queue_counters, restir_pt_initial,
                                            restir_pt_visibility_rays, restir_pt_visibility_results,
                                            restir_visibility_indices, restir_visibility_count);
                                    restir_pt_promote_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                                    std::swap(pt_active, pt_next);
                                }
                                if (has_alpha_visibility || has_transmission) {
                                    launch_restir_visibility_trace<true>(device_scene, paths,
                                        restir_visibility_indices, restir_visibility_count,
                                        restir_pt_visibility_rays, restir_pt_visibility_results,
                                        use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                } else {
                                    launch_restir_visibility_trace<false>(device_scene, paths,
                                        restir_visibility_indices, restir_visibility_count,
                                        restir_pt_visibility_rays, restir_pt_visibility_results,
                                        use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                }
                                restir_pt_spatial_select_kernel<<<full_queue_grid, queue_block_size>>>(
                                    device_scene, restir_pt_current_surfaces, pt_final,
                                    restir_pt_initial, restir_pt_spatial_states,
                                    restir_pt_visibility_results,
                                    restir_pt_history, restir_gi_sequence_index, pixel_count);
                                cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                                restir_pt_inverse_prepare_kernel<GpuRestirPtSpatialState>
                                    <<<full_queue_grid, queue_block_size>>>(
                                        device_scene, settings, restir_pt_current_surfaces, restir_pt_history,
                                        restir_pt_spatial_states, restir_pt_path_states,
                                        restir_pt_visibility_rays, restir_pt_visibility_results,
                                        restir_visibility_indices, restir_visibility_count, pixel_count);
                                restir_pt_reset_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                                pt_active = restir_pt_active_indices;
                                pt_next = restir_pt_next_indices;
                                restir_pt_inverse_replay_setup_kernel<GpuRestirPtSpatialState>
                                    <<<full_queue_grid, queue_block_size>>>(
                                        device_scene, restir_pt_path_states, restir_pt_spatial_states,
                                        restir_pt_history, restir_pt_current_surfaces,
                                        pt_active, restir_pt_queue_counters, pixel_count);
                                for (int replay_step = 0; replay_step < 8; ++replay_step) {
                                    if (has_alpha_visibility) {
                                        launch_restir_pt_trace<true>(device_scene, restir_pt_path_states,
                                            restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                            restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                            full_intersect_grid, intersect_block_size);
                                    } else {
                                        launch_restir_pt_trace<false>(device_scene, restir_pt_path_states,
                                            restir_pt_compact_hits, restir_pt_trace_results, pt_active,
                                            restir_pt_queue_counters, use_two_level, use_wide_bvh, use_cwbvh,
                                            full_intersect_grid, intersect_block_size);
                                    }
                                    restir_pt_replay_resolve_kernel<GpuRestirPtSpatialState>
                                        <<<full_queue_grid, queue_block_size>>>(
                                            device_scene, settings, restir_pt_path_states, restir_pt_spatial_states,
                                            restir_pt_compact_hits, restir_pt_trace_results, pt_active, pt_next,
                                            restir_pt_queue_counters, restir_pt_initial,
                                            restir_pt_visibility_rays, restir_pt_visibility_results,
                                            restir_visibility_indices, restir_visibility_count);
                                    restir_pt_promote_queue_kernel<<<1, 1>>>(restir_pt_queue_counters);
                                    std::swap(pt_active, pt_next);
                                }
                                if (has_alpha_visibility || has_transmission) {
                                    launch_restir_visibility_trace<true>(device_scene, paths,
                                        restir_visibility_indices, restir_visibility_count,
                                        restir_pt_visibility_rays, restir_pt_visibility_results,
                                        use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                } else {
                                    launch_restir_visibility_trace<false>(device_scene, paths,
                                        restir_visibility_indices, restir_visibility_count,
                                        restir_pt_visibility_rays, restir_pt_visibility_results,
                                        use_two_level, use_wide_bvh, use_cwbvh,
                                        full_intersect_grid, intersect_block_size);
                                }
                                restir_pt_pairwise_normalize_kernel<GpuRestirPtSpatialState>
                                    <<<full_queue_grid, queue_block_size>>>(
                                        device_scene, restir_pt_current_surfaces, pt_final,
                                        restir_pt_history, restir_pt_initial, restir_pt_spatial_states,
                                        restir_pt_visibility_results, restir_pt_history, pixel_count);
                                pt_final = restir_pt_history;
                            }
                        }

                        cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                        restir_pt_generate_final_visibility_kernel<<<full_queue_grid, queue_block_size>>>(
                            restir_pt_current_surfaces, pt_final, restir_pt_path_states,
                            restir_pt_visibility_rays, restir_pt_visibility_results,
                            restir_visibility_indices, restir_visibility_count, pixel_count);
                        if (has_alpha_visibility || has_transmission) {
                            launch_restir_visibility_trace<true>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_pt_visibility_rays, restir_pt_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        } else {
                            launch_restir_visibility_trace<false>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_pt_visibility_rays, restir_pt_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        }
                        restir_pt_resolve_final_kernel<<<full_queue_grid, queue_block_size>>>(
                            paths, restir_pt_path_states, pt_final,
                            restir_pt_visibility_results, samples, pixel_count);
                        if (pt_final != restir_pt_history) {
                            cudaMemcpyAsync(restir_pt_history, pt_final,
                                pixels * sizeof(GpuPackedRestirPtReservoir), cudaMemcpyDeviceToDevice);
                        }
                        if (settings.cuda_restir_pt_resampling == RestirPtResamplingMode::TemporalSpatial) {
                            restir_pt_fill_sample_ids_kernel<<<full_queue_grid, queue_block_size>>>(
                                pt_final, restir_pt_sample_ids, pixel_count);
                            const dim3 duplication_block(16, 16);
                            const dim3 duplication_grid(
                                (settings.width + duplication_block.x - 1) / duplication_block.x,
                                (settings.height + duplication_block.y - 1) / duplication_block.y);
                            restir_pt_duplication_map_kernel<<<duplication_grid, duplication_block>>>(
                                restir_pt_sample_ids, restir_pt_duplication_counts,
                                settings.width, settings.height);
                        }
                        cudaMemcpyAsync(restir_pt_history_surfaces, restir_pt_current_surfaces,
                            pixels * sizeof(GpuRestirSurface), cudaMemcpyDeviceToDevice);
                        restir_pt_history_camera_ = scene.camera;
                        restir_pt_history_valid_ = true;
                    } else if (restir_gi_enabled && step == 0) {
                        const ScopedNvtxRange gi_range("ReSTIR GI");
                        restir_gi_generate_secondary_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, bsdf_indices, sample_next_indices,
                            queue_counters, samples, restir_gi_initial_samples, restir_gi_current_surfaces);

                        if (has_alpha_visibility) {
                            launch_restir_gi_secondary_trace<true>(device_scene, settings, paths,
                                bsdf_indices, queue_counters, restir_gi_initial_samples,
                                restir_gi_secondary_compact_hits, restir_gi_secondary_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        } else {
                            launch_restir_gi_secondary_trace<false>(device_scene, settings, paths,
                                bsdf_indices, queue_counters, restir_gi_initial_samples,
                                restir_gi_secondary_compact_hits, restir_gi_secondary_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        }

                        restir_gi_setup_secondary_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, bsdf_indices, queue_counters,
                            restir_gi_secondary_compact_hits, restir_gi_secondary_hits,
                            restir_gi_secondary_results, restir_gi_initial_samples, samples);

                        cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                        #define LT_LAUNCH_RESTIR_GI_DIRECT(LOCAL, ENVIRONMENT) \
                            restir_gi_sample_secondary_direct_kernel<LOCAL, ENVIRONMENT> \
                                <<<full_queue_grid, queue_block_size>>>( \
                                    device_scene, settings, paths, bsdf_indices, queue_counters, \
                                    restir_gi_secondary_hits, restir_gi_secondary_results, \
                                    restir_gi_initial_samples, restir_gi_secondary_direct, \
                                    restir_gi_visibility_rays, restir_gi_visibility_results, \
                                    restir_visibility_indices, restir_visibility_count)
                        if (has_restir_local_lights && has_restir_environment) {
                            LT_LAUNCH_RESTIR_GI_DIRECT(true, true);
                        } else if (has_restir_local_lights) {
                            LT_LAUNCH_RESTIR_GI_DIRECT(true, false);
                        } else if (has_restir_environment) {
                            LT_LAUNCH_RESTIR_GI_DIRECT(false, true);
                        } else {
                            LT_LAUNCH_RESTIR_GI_DIRECT(false, false);
                        }
                        #undef LT_LAUNCH_RESTIR_GI_DIRECT

                        if (has_alpha_visibility || has_transmission) {
                            launch_restir_visibility_trace<true>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_gi_visibility_rays, restir_gi_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        } else {
                            launch_restir_visibility_trace<false>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_gi_visibility_rays, restir_gi_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        }

                        restir_gi_build_initial_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, bsdf_indices, queue_counters,
                            restir_gi_initial_samples, restir_gi_secondary_hits,
                            restir_gi_secondary_results, restir_gi_secondary_direct,
                            restir_gi_visibility_results, restir_gi_initial);

                        GpuPackedRestirGiReservoir* gi_final = restir_gi_initial;
                        if (settings.cuda_restir_gi_resampling != RestirGiResamplingMode::None) {
                            restir_gi_temporal_kernel<<<full_queue_grid, queue_block_size>>>(
                                device_scene, settings, restir_gi_current_surfaces, restir_gi_history_surfaces,
                                restir_gi_initial, restir_gi_history, restir_gi_temporal,
                                restir_gi_temporal_states,
                                restir_gi_history_camera_, restir_gi_history_valid_,
                                restir_gi_sequence_index, pixel_count);
                            restir_gi_temporal_finalize_kernel<<<full_queue_grid, queue_block_size>>>(
                                device_scene, settings, restir_gi_current_surfaces, restir_gi_history_surfaces,
                                restir_gi_initial, restir_gi_temporal_states, restir_gi_temporal, pixel_count);
                            gi_final = restir_gi_temporal;
                            if (settings.cuda_restir_gi_boiling_filter) {
                                const dim3 boiling_block(16, 16);
                                const dim3 boiling_grid(
                                    (settings.width + boiling_block.x - 1) / boiling_block.x,
                                    (settings.height + boiling_block.y - 1) / boiling_block.y);
                                restir_gi_boiling_filter_kernel<<<boiling_grid, boiling_block>>>(
                                    gi_final, restir_gi_scratch, settings.width, settings.height);
                                gi_final = restir_gi_scratch;
                            }
                            if (settings.cuda_restir_gi_resampling == RestirGiResamplingMode::TemporalSpatial) {
                                GpuPackedRestirGiReservoir* spatial_output =
                                    gi_final == restir_gi_scratch ? restir_gi_history : restir_gi_scratch;
                                restir_gi_spatial_kernel<<<full_queue_grid, queue_block_size>>>(
                                    device_scene, settings, restir_gi_current_surfaces,
                                    gi_final, spatial_output, restir_gi_spatial_states,
                                    restir_gi_sequence_index, pixel_count);
                                restir_gi_spatial_finalize_kernel<<<full_queue_grid, queue_block_size>>>(
                                    device_scene, settings, restir_gi_current_surfaces, gi_final,
                                    restir_gi_spatial_states, spatial_output, pixel_count);
                                gi_final = spatial_output;
                            }
                        }

                        cudaMemsetAsync(restir_visibility_count, 0, sizeof(int));
                        restir_gi_generate_final_visibility_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, restir_gi_current_surfaces, gi_final, restir_gi_visibility_rays,
                            restir_gi_visibility_results, restir_gi_initial_samples,
                            restir_visibility_indices, restir_visibility_count, pixel_count);
                        if (has_alpha_visibility || has_transmission) {
                            launch_restir_visibility_trace<true>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_gi_visibility_rays, restir_gi_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        } else {
                            launch_restir_visibility_trace<false>(device_scene, paths,
                                restir_visibility_indices, restir_visibility_count,
                                restir_gi_visibility_rays, restir_gi_visibility_results,
                                use_two_level, use_wide_bvh, use_cwbvh,
                                full_intersect_grid, intersect_block_size);
                        }

                        restir_gi_resolve_final_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, restir_gi_initial_samples,
                            restir_gi_current_surfaces, restir_gi_initial, gi_final,
                            restir_gi_visibility_results, samples, pixel_count);
                        if (gi_final != restir_gi_history) {
                            cudaMemcpyAsync(restir_gi_history, gi_final,
                                pixels * sizeof(GpuPackedRestirGiReservoir), cudaMemcpyDeviceToDevice);
                        }
                        cudaMemcpyAsync(restir_gi_history_surfaces, restir_gi_current_surfaces,
                            pixels * sizeof(GpuRestirSurface), cudaMemcpyDeviceToDevice);
                        restir_gi_history_camera_ = scene.camera;
                        restir_gi_history_valid_ = true;
                    } else {
                        const ScopedNvtxRange range("wavefront BSDF sample");
                        wavefront_bsdf_sample_kernel<<<full_queue_grid, queue_block_size>>>(
                            device_scene, settings, paths, hits, bsdf_indices,
                            sample_next_indices, queue_counters, samples);
                    }
                }

                if (settings.sampling_mode == PathSamplingMode::Unidirectional) {
                    wavefront_bsdf_sample_kernel<<<full_queue_grid, queue_block_size>>>(
                        device_scene, settings, paths, hits, bsdf_indices,
                        sample_next_indices, queue_counters, samples);
                }

                wavefront_promote_next_kernel<<<1, 1>>>(queue_counters);
                std::swap(sample_active_indices, sample_next_indices);
            }
            if (render_error == cudaSuccess) {
                wavefront_finalize_active_kernel<<<full_queue_grid, queue_block_size>>>(
                    paths, sample_active_indices, queue_counters, samples);
                wavefront_accumulate_sample_kernel<<<full_queue_grid, queue_block_size>>>(
                    samples, sample_sum, pixel_count);
            }
        }
        return render_error;
    };
    if (svgf_denoising_enabled(settings)) {
        Vec3* radiance = static_cast<Vec3*>(device_svgf_radiance_);
        Vec3* albedo = static_cast<Vec3*>(device_svgf_albedo_);
        Vec3* emission = static_cast<Vec3*>(device_svgf_emission_);
        Vec3* normal = static_cast<Vec3*>(device_svgf_normal_);
        Vec3* world_position = static_cast<Vec3*>(device_svgf_world_position_);
        float* depth = static_cast<float*>(device_svgf_depth_);
        uint32_t* object_id = static_cast<uint32_t*>(device_svgf_object_id_);
        Vec3* history_illumination = static_cast<Vec3*>(device_svgf_history_illumination_);
        float* history_moment1 = static_cast<float*>(device_svgf_history_moment1_);
        float* history_moment2 = static_cast<float*>(device_svgf_history_moment2_);
        float* history_length = static_cast<float*>(device_svgf_history_length_);
        Vec3* history_normal = static_cast<Vec3*>(device_svgf_history_normal_);
        Vec3* history_world_position = static_cast<Vec3*>(device_svgf_history_world_position_);
        float* history_depth = static_cast<float*>(device_svgf_history_depth_);
        uint32_t* history_object_id = static_cast<uint32_t*>(device_svgf_history_object_id_);
        Vec3* temporal_illumination = static_cast<Vec3*>(device_svgf_temporal_illumination_);
        float* temporal_moment1 = static_cast<float*>(device_svgf_temporal_moment1_);
        float* temporal_moment2 = static_cast<float*>(device_svgf_temporal_moment2_);
        float* temporal_variance = static_cast<float*>(device_svgf_temporal_variance_);
        float* temporal_history_length = static_cast<float*>(device_svgf_temporal_history_length_);
        Vec3* ping_illumination = static_cast<Vec3*>(device_svgf_ping_illumination_);
        Vec3* pong_illumination = static_cast<Vec3*>(device_svgf_pong_illumination_);
        float* ping_variance = static_cast<float*>(device_svgf_ping_variance_);
        float* pong_variance = static_cast<float*>(device_svgf_pong_variance_);
        Vec3* final_color = static_cast<Vec3*>(device_svgf_final_color_);
        Vec3* taa_history_color = static_cast<Vec3*>(device_taa_history_color_);
        Vec3* taa_history_normal = static_cast<Vec3*>(device_taa_history_normal_);
        Vec3* taa_history_world_position = static_cast<Vec3*>(device_taa_history_world_position_);
        float* taa_history_depth = static_cast<float*>(device_taa_history_depth_);
        uint32_t* taa_history_object_id = static_cast<uint32_t*>(device_taa_history_object_id_);
        float* taa_history_length = static_cast<float*>(device_taa_history_length_);

        bool used_rasterized_gbuffer = false;
        if (svgf_gbuffer_interop_.valid) {
            int interop_error_code = static_cast<int>(cudaSuccess);
            used_rasterized_gbuffer = upload_rasterized_gbuffer_interop_to_cuda(
                svgf_gbuffer_interop_,
                settings.width,
                settings.height,
                albedo,
                emission,
                normal,
                world_position,
                depth,
                object_id,
                interop_error_code);
            if (!used_rasterized_gbuffer) {
                std::string key = "CUDA SVGF D3D11 interop unavailable";
                if (interop_error_code != static_cast<int>(cudaSuccess)) {
                    key += ": ";
                    key += cuda_error_text(static_cast<cudaError_t>(interop_error_code));
                }
                if (std::find(reported_fallback_reasons_.begin(), reported_fallback_reasons_.end(), key) == reported_fallback_reasons_.end()) {
                    reported_fallback_reasons_.push_back(key);
                    LT_LOG_WARN("CUDA SVGF using manual traced G-buffer: {}", key);
                } else {
                    LT_LOG_DEBUG("CUDA SVGF using manual traced G-buffer: {}", key);
                }
            }
        }
        if (!wavefront_enabled && !used_rasterized_gbuffer && has_rasterized_svgf_aov(framebuffer)) {
            cuda_error = cudaMemcpy(albedo, framebuffer.aov.albedo.data(), pixels * sizeof(Vec3), cudaMemcpyHostToDevice);
            if (cuda_error == cudaSuccess) {
                cuda_error = cudaMemcpy(emission, framebuffer.aov.emission.data(), pixels * sizeof(Vec3), cudaMemcpyHostToDevice);
            }
            if (cuda_error == cudaSuccess) {
                cuda_error = cudaMemcpy(normal, framebuffer.aov.normal.data(), pixels * sizeof(Vec3), cudaMemcpyHostToDevice);
            }
            if (cuda_error == cudaSuccess) {
                cuda_error = cudaMemcpy(world_position, framebuffer.aov.world_position.data(), pixels * sizeof(Vec3), cudaMemcpyHostToDevice);
            }
            if (cuda_error == cudaSuccess) {
                cuda_error = cudaMemcpy(depth, framebuffer.aov.linear_depth.data(), pixels * sizeof(float), cudaMemcpyHostToDevice);
            }
            if (cuda_error == cudaSuccess) {
                cuda_error = cudaMemcpy(object_id, framebuffer.aov.object_id.data(), pixels * sizeof(uint32_t), cudaMemcpyHostToDevice);
            }
            if (cuda_error != cudaSuccess) {
                reset();
                render_cpu_fallback(scene, settings, framebuffer, "could not upload rasterized CUDA SVGF G-buffer", cuda_error_text(cuda_error));
                return;
            }
            used_rasterized_gbuffer = true;
        }
        // A valid raster G-buffer replaces the manually traced primary AOVs. Upload it
        // before launching the radiance kernel so the kernel can skip that otherwise
        // redundant primary-ray traversal.
        if (wavefront_enabled) {
            Vec3* sample_sum = static_cast<Vec3*>(device_wavefront_sample_sum_);
            const GpuWavefrontSvgfAov svgf_aov = used_rasterized_gbuffer
                ? GpuWavefrontSvgfAov{}
                : GpuWavefrontSvgfAov{albedo, emission, normal, world_position, depth, object_id};
            cuda_error = render_wavefront_samples(sample_sum, true, svgf_aov);
            if (cuda_error == cudaSuccess) {
                const int pixel_count = static_cast<int>(pixels);
                const int queue_block_size = 256;
                const int queue_grid_size = (pixel_count + queue_block_size - 1) / queue_block_size;
                wavefront_svgf_radiance_kernel<<<queue_grid_size, queue_block_size>>>(
                    settings, sample_sum, radiance, pixel_count);
                cuda_error = cudaGetLastError();
            }
            if (cuda_error != cudaSuccess) {
                reset();
                render_cpu_fallback(scene, settings, framebuffer, "CUDA wavefront SVGF input failed", cuda_error_text(cuda_error));
                return;
            }
        } else {
            render_svgf_input_kernel<<<grid, block>>>(
                device_scene,
                settings,
                radiance,
                albedo,
                emission,
                normal,
                world_position,
                depth,
                object_id,
                !used_rasterized_gbuffer);
        }
        const ScopedNvtxRange svgf_range("SVGF");
        {
        const ScopedNvtxRange range("SVGF temporal");
        svgf_temporal_kernel<<<grid, block>>>(
            settings,
            svgf_history_camera_,
            svgf_history_jitter_x_,
            svgf_history_jitter_y_,
            svgf_history_valid_ ? 1 : 0,
            radiance,
            albedo,
            emission,
            normal,
            world_position,
            depth,
            object_id,
            history_illumination,
            history_moment1,
            history_moment2,
            history_length,
            history_normal,
            history_world_position,
            history_depth,
            history_object_id,
            temporal_illumination,
            temporal_moment1,
            temporal_moment2,
            temporal_variance,
            temporal_history_length);
        }
        const int iterations = std::clamp(settings.svgf_iterations, 0, 8);
        Vec3* filtered_illumination = temporal_illumination;
        float* filtered_variance = temporal_variance;
        for (int i = 0; i < iterations; ++i) {
            const ScopedNvtxRange range("SVGF A-Trous");
            Vec3* output_illumination = (i & 1) == 0 ? ping_illumination : pong_illumination;
            float* output_variance = (i & 1) == 0 ? ping_variance : pong_variance;
            svgf_atrous_kernel<<<grid, block>>>(
                settings, 1 << i, filtered_illumination, filtered_variance, normal, depth, object_id,
                output_illumination, output_variance);
            filtered_illumination = output_illumination;
            filtered_variance = output_variance;
        }
        {
        const ScopedNvtxRange range("SVGF resolve");
        svgf_resolve_kernel<<<grid, block>>>(
            settings,
            radiance,
            albedo,
            emission,
            normal,
            world_position,
            depth,
            object_id,
            filtered_illumination,
            filtered_variance,
            temporal_moment1,
            temporal_moment2,
            temporal_history_length,
            history_illumination,
            history_moment1,
            history_moment2,
            history_length,
            history_normal,
            history_world_position,
            history_depth,
            history_object_id,
            final_color);
        }
        taa_resolve_kernel<<<grid, block>>>(
            settings,
            taa_history_camera_,
            taa_history_jitter_x_,
            taa_history_jitter_y_,
            taa_history_valid_ ? 1 : 0,
            final_color,
            normal,
            world_position,
            depth,
            object_id,
            taa_history_color,
            taa_history_normal,
            taa_history_world_position,
            taa_history_depth,
            taa_history_object_id,
            taa_history_length,
            device_rgba);
        const cudaError_t svgf_error = cudaDeviceSynchronize();
        if (svgf_error != cudaSuccess) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "CUDA SVGF kernel failed", cuda_error_text(svgf_error));
            return;
        }
        svgf_history_camera_ = scene.camera;
        svgf_history_jitter_x_ = settings.camera_jitter_x;
        svgf_history_jitter_y_ = settings.camera_jitter_y;
        svgf_history_valid_ = true;
        taa_history_camera_ = scene.camera;
        taa_history_jitter_x_ = settings.camera_jitter_x;
        taa_history_jitter_y_ = settings.camera_jitter_y;
        taa_history_valid_ = true;
    } else {
        cudaError_t render_error = cudaSuccess;
        if (wavefront_enabled) {
            const int pixel_count = static_cast<int>(pixels);
            const int queue_block_size = 256;
            Vec3* sample_sum = static_cast<Vec3*>(device_wavefront_sample_sum_);
            render_error = render_wavefront_samples(sample_sum, false, {});
            if (render_error == cudaSuccess) {
                const int queue_grid_size = (pixel_count + queue_block_size - 1) / queue_block_size;
                wavefront_resolve_kernel<<<queue_grid_size, queue_block_size>>>(
                    settings, sample_sum, device_accumulation, device_rgba, pixel_count);
                render_error = cudaDeviceSynchronize();
            }
        } else {
            render_kernel<<<grid, block>>>(device_scene, settings, device_accumulation, device_rgba);
            render_error = cudaDeviceSynchronize();
        }
        if (render_error != cudaSuccess) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "CUDA kernel failed", cuda_error_text(render_error));
            return;
        }
    }
    const cudaError_t rgba_error = cudaMemcpy(framebuffer.rgba.data(), device_rgba, pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (rgba_error != cudaSuccess) {
        reset();
        framebuffer.clear();
        render_cpu_fallback(scene, settings, framebuffer, "could not copy CUDA RGBA buffer", cuda_error_text(rgba_error));
    }
}

} // namespace lt

#endif

#include "lt/renderer.h"
#include "lt/log.h"

#if LT_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace lt {
namespace {

#include "types.cuh"
#include "math.cuh"
#include "intersection.cuh"
#include "shading.cuh"
#include "kernel.cuh"
#include "scene_upload.cuh"
#include "../cpu/types.inl"
#include "../cpu/intersection.inl"
#include "../cpu/irradiance_volume.inl"
#include "../cpu/shading.inl"
#include "../cpu/irradiance_volume_bake.inl"

struct PackedGpuIrradianceVolume {
    GpuIrradianceVolume volume;
    std::vector<Vec3> directions;
    std::vector<Vec3> irradiance;
    std::vector<GpuIrradianceVolumeGrid> grids;
    std::vector<int> cell_subgrid_indices;
    std::vector<GpuIrradianceVolumeDebugProbe> debug_probes;
};

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

const char* cuda_error_text(cudaError_t error) {
    return error == cudaSuccess ? nullptr : cudaGetErrorString(error);
}

} // namespace

bool CudaPathTracer::available() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaPathTracer::~CudaPathTracer() {
    reset();
}

void CudaPathTracer::reset() {
    cudaFree(device_accumulation_);
    cudaFree(device_rgba_);
    cudaFree(device_scene_);
    cudaFree(device_materials_);
    cudaFree(device_textures_);
    cudaFree(device_triangles_);
    cudaFree(device_spheres_);
    cudaFree(device_triangle_indices_);
    cudaFree(device_light_indices_);
    cudaFree(device_directional_lights_);
    cudaFree(device_bvh_nodes_);
    cudaFree(device_mesh_instances_);
    cudaFree(device_mesh_instance_indices_);
    cudaFree(device_tlas_nodes_);
    cudaFree(device_irradiance_volume_directions_);
    cudaFree(device_irradiance_volume_irradiance_);
    cudaFree(device_irradiance_volume_grids_);
    cudaFree(device_irradiance_volume_cells_);
    cudaFree(device_irradiance_volume_debug_probes_);
    device_accumulation_ = nullptr;
    device_rgba_ = nullptr;
    device_scene_ = nullptr;
    device_materials_ = nullptr;
    device_textures_ = nullptr;
    device_triangles_ = nullptr;
    device_spheres_ = nullptr;
    device_triangle_indices_ = nullptr;
    device_light_indices_ = nullptr;
    device_directional_lights_ = nullptr;
    device_bvh_nodes_ = nullptr;
    device_mesh_instances_ = nullptr;
    device_mesh_instance_indices_ = nullptr;
    device_tlas_nodes_ = nullptr;
    device_irradiance_volume_directions_ = nullptr;
    device_irradiance_volume_irradiance_ = nullptr;
    device_irradiance_volume_grids_ = nullptr;
    device_irradiance_volume_cells_ = nullptr;
    device_irradiance_volume_debug_probes_ = nullptr;
    cached_render_scene_ = {};
    cached_irradiance_volume_.reset();
    cached_pixels_ = 0;
    cached_materials_ = 0;
    cached_textures_ = 0;
    cached_triangles_ = 0;
    cached_spheres_ = 0;
    cached_triangle_indices_ = 0;
    cached_lights_ = 0;
    cached_directional_lights_ = 0;
    cached_bvh_nodes_ = 0;
    cached_mesh_instances_ = 0;
    cached_mesh_instance_indices_ = 0;
    cached_tlas_nodes_ = 0;
    cached_irradiance_volume_directions_ = 0;
    cached_irradiance_volume_irradiance_ = 0;
    cached_irradiance_volume_grids_ = 0;
    cached_irradiance_volume_cells_ = 0;
    cached_irradiance_volume_debug_probes_ = 0;
    scene_uploaded_ = false;
    cached_render_scene_valid_ = false;
    cached_irradiance_volume_enabled_ = false;
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

void CudaPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    const size_t pixels = static_cast<size_t>(settings.width) * static_cast<size_t>(settings.height);

    if (cached_pixels_ != pixels) {
        reset();
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
    cached_pixels_ = pixels;

    Vec3* device_accumulation = static_cast<Vec3*>(device_accumulation_);
    uint32_t* device_rgba = static_cast<uint32_t*>(device_rgba_);
    GpuScene* device_scene = static_cast<GpuScene*>(device_scene_);

    if (has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Transform)) {
        cached_render_scene_valid_ = false;
    }

    std::shared_ptr<IrradianceVolume> irradiance_volume;
    bool irradiance_volume_rebuilt = false;
    const bool irradiance_volume_enabled_changed = cached_irradiance_volume_enabled_ != settings.use_irradiance_volume;
    if (irradiance_volume_rendering_enabled(settings)) {
        if (!cached_render_scene_valid_) {
            cached_render_scene_ = build_render_scene(scene);
            cached_render_scene_valid_ = true;
        }
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
        has_dirty(settings.dirty, RenderDirty::Environment);
    const bool irradiance_volume_upload = full_upload || irradiance_volume_rebuilt || irradiance_volume_enabled_changed;
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

    if (full_upload) {
        PackedGpuScene packed;
        if (!pack_scene(scene, settings, packed)) {
            render_cpu_fallback(scene, settings, framebuffer, "could not pack scene for CUDA");
            return;
        }
        const bool upload_textures = texture_objects_.size() != packed.textures.size() || has_dirty(settings.dirty, RenderDirty::Texture);
        if (upload_textures ? !upload_texture_objects(scene, packed, texture_arrays_, texture_objects_) : !apply_cached_texture_objects(packed, texture_objects_)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, upload_textures ? "could not upload CUDA texture objects" : "could not apply cached CUDA texture objects");
            return;
        }
        if (!upload_buffer(device_materials_, cached_materials_, packed.materials) ||
            !upload_buffer(device_textures_, cached_textures_, packed.textures) ||
            !upload_buffer(device_triangles_, cached_triangles_, packed.triangles) ||
            !upload_buffer(device_spheres_, cached_spheres_, packed.spheres) ||
            !upload_buffer(device_triangle_indices_, cached_triangle_indices_, packed.triangle_indices) ||
            !upload_buffer(device_light_indices_, cached_lights_, packed.light_indices) ||
            !upload_buffer(device_directional_lights_, cached_directional_lights_, packed.directional_lights) ||
            !upload_buffer(device_bvh_nodes_, cached_bvh_nodes_, packed.bvh_nodes) ||
            !upload_buffer(device_mesh_instances_, cached_mesh_instances_, packed.mesh_instances) ||
            !upload_buffer(device_mesh_instance_indices_, cached_mesh_instance_indices_, packed.mesh_instance_indices) ||
            !upload_buffer(device_tlas_nodes_, cached_tlas_nodes_, packed.tlas_nodes)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA scene buffers");
            return;
        }
        packed.scene.materials = static_cast<GpuMaterial*>(device_materials_);
        packed.scene.textures = static_cast<GpuTexture*>(device_textures_);
        packed.scene.triangles = static_cast<GpuTriangle*>(device_triangles_);
        packed.scene.spheres = static_cast<GpuSphere*>(device_spheres_);
        packed.scene.triangle_indices = static_cast<int*>(device_triangle_indices_);
        packed.scene.light_indices = static_cast<int*>(device_light_indices_);
        packed.scene.directional_lights = static_cast<GpuDirectionalLight*>(device_directional_lights_);
        packed.scene.bvh_nodes = static_cast<GpuBvhNode*>(device_bvh_nodes_);
        packed.scene.mesh_instances = static_cast<GpuMeshInstance*>(device_mesh_instances_);
        packed.scene.mesh_instance_indices = static_cast<int*>(device_mesh_instance_indices_);
        packed.scene.tlas_nodes = static_cast<GpuBvhNode*>(device_tlas_nodes_);
        if (!upload_gpu_irradiance_volume(irradiance_volume.get(), packed.scene.irradiance_volume)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA irradiance volume");
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
    } else {
        if (has_dirty(settings.dirty, RenderDirty::Camera) && !upload_camera(device_scene, scene.camera)) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA camera");
            return;
        }
        if (has_dirty(settings.dirty, RenderDirty::Environment)) {
            GpuScene environment_scene;
            if (!make_environment_gpu(scene, environment_scene) || !upload_environment(device_scene, environment_scene)) {
                reset();
                render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA environment");
                return;
            }
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
    }

    if (settings.frame_index == 0u || has_dirty(settings.dirty, RenderDirty::Render)) {
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
    render_kernel<<<grid, block>>>(device_scene, settings, device_accumulation, device_rgba);
    const cudaError_t render_error = cudaDeviceSynchronize();
    if (render_error != cudaSuccess) {
        reset();
        render_cpu_fallback(scene, settings, framebuffer, "CUDA kernel failed", cuda_error_text(render_error));
        return;
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

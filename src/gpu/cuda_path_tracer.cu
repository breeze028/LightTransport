#include "lt/renderer.h"
#include "lt/log.h"

#if LT_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
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
    cudaFree(device_bvh_nodes_);
    cudaFree(device_mesh_instances_);
    cudaFree(device_mesh_instance_indices_);
    cudaFree(device_tlas_nodes_);
    device_accumulation_ = nullptr;
    device_rgba_ = nullptr;
    device_scene_ = nullptr;
    device_materials_ = nullptr;
    device_textures_ = nullptr;
    device_triangles_ = nullptr;
    device_spheres_ = nullptr;
    device_triangle_indices_ = nullptr;
    device_light_indices_ = nullptr;
    device_bvh_nodes_ = nullptr;
    device_mesh_instances_ = nullptr;
    device_mesh_instance_indices_ = nullptr;
    device_tlas_nodes_ = nullptr;
    cached_pixels_ = 0;
    cached_materials_ = 0;
    cached_textures_ = 0;
    cached_triangles_ = 0;
    cached_spheres_ = 0;
    cached_triangle_indices_ = 0;
    cached_lights_ = 0;
    cached_bvh_nodes_ = 0;
    cached_mesh_instances_ = 0;
    cached_mesh_instance_indices_ = 0;
    cached_tlas_nodes_ = 0;
    scene_uploaded_ = false;
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

    const bool full_upload = !scene_uploaded_ ||
        has_dirty(settings.dirty, RenderDirty::Geometry) ||
        has_dirty(settings.dirty, RenderDirty::Material) ||
        has_dirty(settings.dirty, RenderDirty::Texture);

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
        packed.scene.bvh_nodes = static_cast<GpuBvhNode*>(device_bvh_nodes_);
        packed.scene.mesh_instances = static_cast<GpuMeshInstance*>(device_mesh_instances_);
        packed.scene.mesh_instance_indices = static_cast<int*>(device_mesh_instance_indices_);
        packed.scene.tlas_nodes = static_cast<GpuBvhNode*>(device_tlas_nodes_);
        cuda_error = cudaMemcpy(device_scene, &packed.scene, sizeof(GpuScene), cudaMemcpyHostToDevice);
        if (cuda_error != cudaSuccess) {
            reset();
            render_cpu_fallback(scene, settings, framebuffer, "could not upload CUDA scene header", cuda_error_text(cuda_error));
            return;
        }
        scene_uploaded_ = true;
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

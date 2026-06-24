#include "lt/renderer.h"

#if LT_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace lt {
namespace {

#include "types.cuh"
#include "math.cuh"
#include "intersection.cuh"
#include "shading.cuh"
#include "kernel.cuh"
#include "scene_upload.cuh"

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

void CudaPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    framebuffer.resize(settings.width, settings.height);
    const size_t pixels = static_cast<size_t>(settings.width) * static_cast<size_t>(settings.height);

    if (cached_pixels_ != pixels) {
        reset();
    }

    if (!device_accumulation_ && cudaMalloc(&device_accumulation_, pixels * sizeof(Vec3)) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    if (!device_rgba_ && cudaMalloc(&device_rgba_, pixels * sizeof(uint32_t)) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    if (!device_scene_ && cudaMalloc(&device_scene_, sizeof(GpuScene)) != cudaSuccess) {
        reset();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
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
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        const bool upload_textures = texture_objects_.size() != packed.textures.size() || has_dirty(settings.dirty, RenderDirty::Texture);
        if (upload_textures ? !upload_texture_objects(scene, packed, texture_arrays_, texture_objects_) : !apply_cached_texture_objects(packed, texture_objects_)) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
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
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
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
        if (cudaMemcpy(device_scene, &packed.scene, sizeof(GpuScene), cudaMemcpyHostToDevice) != cudaSuccess) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        scene_uploaded_ = true;
    } else {
        if (has_dirty(settings.dirty, RenderDirty::Camera) && !upload_camera(device_scene, scene.camera)) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
            return;
        }
        if (has_dirty(settings.dirty, RenderDirty::Environment)) {
            GpuScene environment_scene;
            if (!make_environment_gpu(scene, environment_scene) || !upload_environment(device_scene, environment_scene)) {
                reset();
                CpuPathTracer fallback;
                fallback.render(scene, settings, framebuffer);
                return;
            }
        }
    }

    if (settings.frame_index == 0u || has_dirty(settings.dirty, RenderDirty::Render)) {
        if (cudaMemset(device_accumulation, 0, pixels * sizeof(Vec3)) != cudaSuccess) {
            reset();
            CpuPathTracer fallback;
            fallback.render(scene, settings, framebuffer);
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
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
        return;
    }
    const cudaError_t rgba_error = cudaMemcpy(framebuffer.rgba.data(), device_rgba, pixels * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (rgba_error != cudaSuccess) {
        reset();
        framebuffer.clear();
        CpuPathTracer fallback;
        fallback.render(scene, settings, framebuffer);
    }
}

} // namespace lt

#endif

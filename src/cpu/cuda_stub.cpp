#include "lt/renderer.h"
#include "lt/log.h"

#if !LT_HAS_CUDA

#include <algorithm>
#include <string>

namespace lt {

CudaPathTracer::~CudaPathTracer() = default;
bool CudaPathTracer::available() const { return false; }
void CudaPathTracer::reset() {}
void CudaPathTracer::render_cpu_fallback(
    const Scene& scene,
    const RenderSettings& settings,
    Framebuffer& framebuffer,
    const char* reason,
    const char*) {
    const std::string key = reason ? reason : "CUDA backend is not built";
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
    render_cpu_fallback(scene, settings, framebuffer, "CUDA backend is not built");
}

void CudaPathTracer::render_with_rasterized_gbuffer_interop(
    const Scene& scene,
    const RenderSettings& settings,
    Framebuffer& framebuffer,
    const RasterizedGBufferInterop&) {
    render_cpu_fallback(scene, settings, framebuffer, "CUDA backend is not built");
}

std::shared_ptr<void> build_irradiance_volume_gpu(
    const RenderScene&, const Scene&, const RenderSettings&) {
    LT_LOG_WARN("GPU irradiance volume bake requested but CUDA is not available. "
                "Rebuild with CUDA support or switch to CPU bake backend.");
    return nullptr;
}

std::shared_ptr<void> build_lightmap_gpu(
    const RenderScene&, const Scene&, const RenderSettings&) {
    LT_LOG_WARN("GPU lightmap bake requested but CUDA is not available. "
                "Rebuild with CUDA support or switch to CPU bake backend.");
    return nullptr;
}

} // namespace lt

#endif

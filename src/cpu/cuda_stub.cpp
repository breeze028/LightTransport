#include "lt/renderer.h"

#if !LT_HAS_CUDA

namespace lt {

CudaPathTracer::~CudaPathTracer() = default;
bool CudaPathTracer::available() const { return false; }
void CudaPathTracer::reset() {}
void CudaPathTracer::render(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    CpuPathTracer fallback;
    fallback.render(scene, settings, framebuffer);
}

} // namespace lt

#endif

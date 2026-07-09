#pragma once

#include "lt/renderer.h"

namespace lt {

void apply_svgf_denoiser(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer);

} // namespace lt

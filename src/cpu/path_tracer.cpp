#include "lt/renderer.h"
#include "lt/log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace lt {
namespace {

#include "types.inl"
#include "camera.inl"
#include "intersection.inl"
#include "irradiance_volume.inl"
#include "shading.inl"
#include "irradiance_volume_bake.inl"

} // namespace

#include "renderer.inl"

} // namespace lt

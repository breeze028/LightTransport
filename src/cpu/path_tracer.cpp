#include "lt/renderer.h"
#include "lt/log.h"

#include <xatlas.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <thread>
#include <vector>

namespace lt {
namespace {

#include "types.inl"
#include "camera.inl"
#include "intersection.inl"
#include "irradiance_volume.inl"
#include "lightmap.inl"
#include "shading.inl"
#include "irradiance_volume_bake.inl"
#include "lightmap_bake.inl"

} // namespace

#include "renderer.inl"

} // namespace lt

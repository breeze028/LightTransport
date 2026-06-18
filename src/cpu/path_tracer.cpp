#include "lt/renderer.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace lt {
namespace {

#include "types.inl"
#include "camera.inl"
#include "intersection.inl"
#include "shading.inl"

} // namespace

#include "renderer.inl"

} // namespace lt

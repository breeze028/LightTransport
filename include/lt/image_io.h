#pragma once

#include "lt/renderer.h"

#include <string>

namespace lt {

bool write_ppm(const std::string& path, const Framebuffer& framebuffer, std::string* error = nullptr);
bool write_png(const std::string& path, const Framebuffer& framebuffer, std::string* error = nullptr);
bool write_image(const std::string& path, const Framebuffer& framebuffer, std::string* error = nullptr);

} // namespace lt

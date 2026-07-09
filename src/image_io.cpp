#include "lt/image_io.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "stb_image_write.h"

namespace lt {
namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

bool valid_framebuffer(const Framebuffer& framebuffer, std::string* error) {
    if (framebuffer.width <= 0 || framebuffer.height <= 0) {
        set_error(error, "Framebuffer has invalid dimensions");
        return false;
    }
    const size_t expected = static_cast<size_t>(framebuffer.width) * static_cast<size_t>(framebuffer.height);
    if (framebuffer.rgba.size() != expected) {
        set_error(error, "Framebuffer RGBA buffer size does not match dimensions");
        return false;
    }
    return true;
}

std::string lowercase_extension(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

std::vector<uint8_t> framebuffer_to_rgba8(const Framebuffer& framebuffer) {
    std::vector<uint8_t> pixels;
    pixels.resize(framebuffer.rgba.size() * 4u);
    for (size_t i = 0; i < framebuffer.rgba.size(); ++i) {
        const uint32_t color = framebuffer.rgba[i];
        pixels[i * 4u + 0u] = static_cast<uint8_t>((color >> 16u) & 0xffu);
        pixels[i * 4u + 1u] = static_cast<uint8_t>((color >> 8u) & 0xffu);
        pixels[i * 4u + 2u] = static_cast<uint8_t>(color & 0xffu);
        pixels[i * 4u + 3u] = static_cast<uint8_t>((color >> 24u) & 0xffu);
    }
    return pixels;
}

} // namespace

bool write_ppm(const std::string& path, const Framebuffer& framebuffer, std::string* error) {
    if (!valid_framebuffer(framebuffer, error)) {
        return false;
    }
    std::ofstream output(path);
    if (!output) {
        set_error(error, "Could not open output image: " + path);
        return false;
    }
    output << "P3\n" << framebuffer.width << ' ' << framebuffer.height << "\n255\n";
    for (uint32_t color : framebuffer.rgba) {
        output << ((color >> 16u) & 0xffu) << ' '
            << ((color >> 8u) & 0xffu) << ' '
            << (color & 0xffu) << '\n';
    }
    if (!output) {
        set_error(error, "Could not finish writing output image: " + path);
        return false;
    }
    return true;
}

bool write_png(const std::string& path, const Framebuffer& framebuffer, std::string* error) {
    if (!valid_framebuffer(framebuffer, error)) {
        return false;
    }
    const std::vector<uint8_t> pixels = framebuffer_to_rgba8(framebuffer);
    const int stride = framebuffer.width * 4;
    if (stbi_write_png(path.c_str(), framebuffer.width, framebuffer.height, 4, pixels.data(), stride) == 0) {
        set_error(error, "Could not write PNG image: " + path);
        return false;
    }
    return true;
}

bool write_image(const std::string& path, const Framebuffer& framebuffer, std::string* error) {
    const std::string ext = lowercase_extension(path);
    if (ext == ".png") {
        return write_png(path, framebuffer, error);
    }
    if (ext == ".ppm" || ext.empty()) {
        return write_ppm(path, framebuffer, error);
    }
    set_error(error, "Unsupported image extension '" + ext + "'. Expected .png or .ppm");
    return false;
}

} // namespace lt

#include "lt/texture.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

namespace lt {
namespace {

float wrap01(float v) {
    v = v - std::floor(v);
    return v < 0.0f ? v + 1.0f : v;
}

bool read_token(std::istream& in, std::string& token) {
    token.clear();
    char c = 0;
    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (c == '#') {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        token.push_back(c);
        break;
    }
    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            break;
        }
        token.push_back(c);
    }
    return !token.empty();
}

} // namespace

Vec3 Texture::sample(Vec2 uv) const {
    if (width <= 0 || height <= 0 || pixels.empty()) {
        return {1.0f, 1.0f, 1.0f};
    }
    const float u = wrap01(uv.x);
    const float v = wrap01(uv.y);
    const int x = std::clamp(static_cast<int>(u * static_cast<float>(width)), 0, width - 1);
    const int y = std::clamp(static_cast<int>((1.0f - v) * static_cast<float>(height)), 0, height - 1);
    return pixels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
}

bool load_ppm_texture(const std::string& name, const std::string& path, Texture& texture, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Could not open texture: " + path;
        return false;
    }

    std::string token;
    if (!read_token(in, token) || (token != "P3" && token != "P6")) {
        error = "Texture must be PPM P3/P6: " + path;
        return false;
    }
    const bool binary = token == "P6";
    if (!read_token(in, token)) return false;
    const int width = std::max(0, std::stoi(token));
    if (!read_token(in, token)) return false;
    const int height = std::max(0, std::stoi(token));
    if (!read_token(in, token)) return false;
    const int max_value = std::max(1, std::stoi(token));
    if (width <= 0 || height <= 0) {
        error = "Texture has invalid dimensions: " + path;
        return false;
    }

    texture = {};
    texture.name = name;
    texture.path = path;
    texture.width = width;
    texture.height = height;
    texture.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    const float inv = 1.0f / static_cast<float>(max_value);

    if (binary) {
        std::vector<unsigned char> bytes(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
            error = "Texture ended early: " + path;
            return false;
        }
        for (size_t i = 0; i < texture.pixels.size(); ++i) {
            texture.pixels[i] = {
                static_cast<float>(bytes[i * 3u + 0u]) * inv,
                static_cast<float>(bytes[i * 3u + 1u]) * inv,
                static_cast<float>(bytes[i * 3u + 2u]) * inv,
            };
        }
    } else {
        for (Vec3& pixel : texture.pixels) {
            std::string r;
            std::string g;
            std::string b;
            if (!read_token(in, r) || !read_token(in, g) || !read_token(in, b)) {
                error = "Texture ended early: " + path;
                return false;
            }
            pixel = {std::stof(r) * inv, std::stof(g) * inv, std::stof(b) * inv};
        }
    }
    return true;
}

} // namespace lt

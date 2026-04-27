#pragma once

#include "lt/math.h"

#include <string>
#include <vector>

namespace lt {

struct Texture {
    std::string name;
    std::string path;
    int width = 0;
    int height = 0;
    std::vector<Vec3> pixels;

    Vec3 sample(Vec2 uv) const;
};

bool load_ppm_texture(const std::string& name, const std::string& path, Texture& texture, std::string& error);

} // namespace lt

#pragma once

#include "lt/math.h"

#include <cstddef>
#include <string>
#include <vector>

namespace lt {

enum class TextureWrap2D {
    Repeat,
    RepeatClampY,
    OctahedralSphere,
};

struct Texture {
    std::string name;
    std::string path;
    int width = 0;
    int height = 0;
    std::vector<Vec3> pixels;
    std::vector<float> alpha;
    std::vector<int> mip_widths;
    std::vector<int> mip_heights;
    std::vector<std::vector<Vec3>> mip_pixels;

    Vec3 sample(Vec2 uv) const;
    Vec3 sample_lod(Vec2 uv, float lod) const;
    Vec3 sample_lod(Vec2 uv, float lod, TextureWrap2D wrap) const;
    float sample_alpha(Vec2 uv) const;
    void build_mips();
};

bool load_ppm_texture(const std::string& name, const std::string& path, Texture& texture, std::string& error);
bool load_hdr_texture(const std::string& name, const std::string& path, Texture& texture, std::string& error);
bool load_exr_texture(const std::string& name, const std::string& path, Texture& texture, std::string& error);
bool load_texture_file(const std::string& name, const std::string& path, Texture& texture, std::string& error);
bool load_texture_memory(const std::string& name, const unsigned char* data, size_t size, Texture& texture, std::string& error);

} // namespace lt

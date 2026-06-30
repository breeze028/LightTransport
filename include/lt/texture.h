#pragma once

#include "lt/math.h"

#include <cstddef>
#include <string>
#include <vector>

namespace lt {

enum class TextureRole {
    Unknown = 0,
    Color = 1,
    Data = 2,
    Normal = 3,
    Emission = 4,
    Environment = 5,
};

enum class TextureColorSpace {
    Auto = 0,
    SceneLinear = 1,
    SRGB = 2,
    Raw = 3,
};

enum class TextureWrap2D {
    Repeat,
    RepeatClampY,
    OctahedralSphere,
};

struct Texture {
    std::string name;
    std::string path;
    TextureRole role = TextureRole::Unknown;
    TextureColorSpace color_space = TextureColorSpace::Auto;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> encoded_bytes;
    std::string encoded_extension;
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
void apply_texture_role(Texture& texture, TextureRole role, TextureColorSpace color_space);
bool write_texture_png(const Texture& texture, const std::string& path, std::string& error);
bool write_texture_hdr(const Texture& texture, const std::string& path, std::string& error);

} // namespace lt

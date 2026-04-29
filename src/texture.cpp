#include "lt/texture.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <objbase.h>
#include <wincodec.h>
#endif

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

std::string lowercase_extension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? std::string{} : path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

float rgbe_channel(unsigned char value, unsigned char exponent) {
    return exponent ? std::ldexp(static_cast<float>(value) / 256.0f, static_cast<int>(exponent) - 128) : 0.0f;
}

Vec3 rgbe_to_vec3(const unsigned char rgbe[4]) {
    return {
        rgbe_channel(rgbe[0], rgbe[3]),
        rgbe_channel(rgbe[1], rgbe[3]),
        rgbe_channel(rgbe[2], rgbe[3]),
    };
}

bool read_hdr_scanline(std::istream& in, int width, std::vector<Vec3>& pixels, int y) {
    unsigned char header[4] = {};
    if (!in.read(reinterpret_cast<char*>(header), 4)) {
        return false;
    }
    if (width < 8 || width > 32767 || header[0] != 2 || header[1] != 2 || (header[2] & 0x80u)) {
        unsigned char rgbe[4] = {header[0], header[1], header[2], header[3]};
        pixels[static_cast<size_t>(y) * static_cast<size_t>(width)] = rgbe_to_vec3(rgbe);
        for (int x = 1; x < width; ++x) {
            if (!in.read(reinterpret_cast<char*>(rgbe), 4)) {
                return false;
            }
            pixels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = rgbe_to_vec3(rgbe);
        }
        return true;
    }
    const int scanline_width = (static_cast<int>(header[2]) << 8) | static_cast<int>(header[3]);
    if (scanline_width != width) {
        return false;
    }

    std::vector<unsigned char> channels(static_cast<size_t>(width) * 4u);
    for (int channel = 0; channel < 4; ++channel) {
        int x = 0;
        while (x < width) {
            unsigned char code = 0;
            if (!in.read(reinterpret_cast<char*>(&code), 1)) {
                return false;
            }
            if (code > 128) {
                const int count = code - 128;
                unsigned char value = 0;
                if (count <= 0 || x + count > width || !in.read(reinterpret_cast<char*>(&value), 1)) {
                    return false;
                }
                for (int i = 0; i < count; ++i) {
                    channels[static_cast<size_t>(channel) * static_cast<size_t>(width) + static_cast<size_t>(x++)] = value;
                }
            } else {
                const int count = code;
                if (count <= 0 || x + count > width) {
                    return false;
                }
                if (!in.read(reinterpret_cast<char*>(channels.data() + static_cast<size_t>(channel) * static_cast<size_t>(width) + static_cast<size_t>(x)), count)) {
                    return false;
                }
                x += count;
            }
        }
    }
    for (int x = 0; x < width; ++x) {
        const unsigned char rgbe[4] = {
            channels[static_cast<size_t>(x)],
            channels[static_cast<size_t>(width) + static_cast<size_t>(x)],
            channels[static_cast<size_t>(width) * 2u + static_cast<size_t>(x)],
            channels[static_cast<size_t>(width) * 3u + static_cast<size_t>(x)],
        };
        pixels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] = rgbe_to_vec3(rgbe);
    }
    return true;
}

#if defined(_WIN32)
std::wstring widen(const std::string& text) {
    if (text.empty()) {
        return {};
    }
    const int needed = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (needed <= 0) {
        return std::wstring(text.begin(), text.end());
    }
    std::wstring result(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, result.data(), needed);
    return result;
}

template <typename T>
void release_com(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

bool decode_wic_frame(IWICBitmapSource* source, const std::string& name, Texture& texture, std::string& error) {
    IWICImagingFactory* factory = nullptr;
    IWICFormatConverter* converter = nullptr;
    UINT width = 0;
    UINT height = 0;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) {
        hr = source->GetSize(&width, &height);
    }
    if (SUCCEEDED(hr)) {
        hr = factory->CreateFormatConverter(&converter);
    }
    if (SUCCEEDED(hr)) {
        hr = converter->Initialize(source, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
    }
    if (FAILED(hr) || width == 0 || height == 0) {
        error = "Could not decode texture image";
        release_com(converter);
        release_com(factory);
        return false;
    }

    std::vector<unsigned char> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    hr = converter->CopyPixels(nullptr, width * 4u, static_cast<UINT>(rgba.size()), rgba.data());
    release_com(converter);
    release_com(factory);
    if (FAILED(hr)) {
        error = "Could not read texture pixels";
        return false;
    }

    texture = {};
    texture.name = name;
    texture.width = static_cast<int>(width);
    texture.height = static_cast<int>(height);
    texture.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (size_t i = 0; i < texture.pixels.size(); ++i) {
        texture.pixels[i] = {
            static_cast<float>(rgba[i * 4u + 0u]) / 255.0f,
            static_cast<float>(rgba[i * 4u + 1u]) / 255.0f,
            static_cast<float>(rgba[i * 4u + 2u]) / 255.0f,
        };
    }
    return true;
}

bool load_wic_file(const std::string& name, const std::string& path, Texture& texture, std::string& error) {
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninit = SUCCEEDED(init);
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    const std::wstring wide_path = widen(path);
    if (SUCCEEDED(hr)) {
        hr = factory->CreateDecoderFromFilename(wide_path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder);
    }
    if (SUCCEEDED(hr)) {
        hr = decoder->GetFrame(0, &frame);
    }
    bool ok = false;
    if (SUCCEEDED(hr)) {
        ok = decode_wic_frame(frame, name, texture, error);
        texture.path = path;
    } else {
        error = "Could not open texture: " + path;
    }
    release_com(frame);
    release_com(decoder);
    release_com(factory);
    if (uninit) {
        CoUninitialize();
    }
    return ok;
}

bool load_wic_memory(const std::string& name, const unsigned char* data, size_t size, Texture& texture, std::string& error) {
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninit = SUCCEEDED(init);
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) {
        hr = factory->CreateStream(&stream);
    }
    if (SUCCEEDED(hr)) {
        hr = stream->InitializeFromMemory(const_cast<BYTE*>(reinterpret_cast<const BYTE*>(data)), static_cast<DWORD>(size));
    }
    if (SUCCEEDED(hr)) {
        hr = factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnLoad, &decoder);
    }
    if (SUCCEEDED(hr)) {
        hr = decoder->GetFrame(0, &frame);
    }
    bool ok = false;
    if (SUCCEEDED(hr)) {
        ok = decode_wic_frame(frame, name, texture, error);
    } else {
        error = "Could not decode embedded texture";
    }
    release_com(frame);
    release_com(decoder);
    release_com(stream);
    release_com(factory);
    if (uninit) {
        CoUninitialize();
    }
    return ok;
}
#endif

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

bool load_hdr_texture(const std::string& name, const std::string& path, Texture& texture, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Could not open HDR texture: " + path;
        return false;
    }

    std::string line;
    int width = 0;
    int height = 0;
    bool found_format = false;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.rfind("FORMAT=", 0) == 0) {
            found_format = line.find("32-bit_rle_rgbe") != std::string::npos;
            continue;
        }
        if (line.find("-Y") != std::string::npos || line.find("+Y") != std::string::npos) {
            std::istringstream size_line(line);
            std::string y_token;
            std::string x_token;
            size_line >> y_token >> height >> x_token >> width;
            break;
        }
    }
    if (!found_format || width <= 0 || height <= 0) {
        error = "Invalid Radiance HDR texture: " + path;
        return false;
    }

    texture = {};
    texture.name = name;
    texture.path = path;
    texture.width = width;
    texture.height = height;
    texture.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        if (!read_hdr_scanline(in, width, texture.pixels, y)) {
            error = "HDR texture ended early: " + path;
            return false;
        }
    }
    return true;
}

bool load_texture_file(const std::string& name, const std::string& path, Texture& texture, std::string& error) {
    const std::string ext = lowercase_extension(path);
    if (ext == ".hdr") {
        return load_hdr_texture(name, path, texture, error);
    }
    if (ext == ".ppm") {
        return load_ppm_texture(name, path, texture, error);
    }
#if defined(_WIN32)
    return load_wic_file(name, path, texture, error);
#else
    error = "Only PPM textures are supported on this platform: " + path;
    return false;
#endif
}

bool load_texture_memory(const std::string& name, const unsigned char* data, size_t size, Texture& texture, std::string& error) {
    if (!data || size == 0) {
        error = "Embedded texture is empty";
        return false;
    }
#if defined(_WIN32)
    return load_wic_memory(name, data, size, texture, error);
#else
    error = "Embedded PNG/JPEG textures are only supported on Windows builds";
    return false;
#endif
}

} // namespace lt

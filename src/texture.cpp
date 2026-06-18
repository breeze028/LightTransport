#include "lt/texture.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <unordered_map>

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

Vec3 lerp_vec3(Vec3 a, Vec3 b, float t) {
    return a * (1.0f - t) + b * t;
}

Vec2 wrap_oct_sphere(Vec2 uv) {
    float u = uv.x;
    float v = uv.y;
    if (u < 0.0f) {
        u = -u;
        v = 1.0f - v;
    } else if (u > 1.0f) {
        u = 2.0f - u;
        v = 1.0f - v;
    }
    if (v < 0.0f) {
        v = -v;
        u = 1.0f - u;
    } else if (v > 1.0f) {
        v = 2.0f - v;
        u = 1.0f - u;
    }
    return {std::clamp(u, 0.0f, 1.0f), std::clamp(v, 0.0f, 1.0f)};
}

Vec3 sample_level(const std::vector<Vec3>& pixels, int width, int height, Vec2 uv, TextureWrap2D wrap = TextureWrap2D::Repeat) {
    if (width <= 0 || height <= 0 || pixels.empty()) {
        return {1.0f, 1.0f, 1.0f};
    }
    if (wrap == TextureWrap2D::OctahedralSphere) {
        uv = wrap_oct_sphere(uv);
    }
    const float su = wrap == TextureWrap2D::Repeat || wrap == TextureWrap2D::RepeatClampY ? wrap01(uv.x) : std::clamp(uv.x, 0.0f, 1.0f);
    const float sv = wrap == TextureWrap2D::Repeat ? wrap01(uv.y) : std::clamp(uv.y, 0.0f, 1.0f);
    const float x = su * static_cast<float>(width) - 0.5f;
    const float image_v = wrap == TextureWrap2D::OctahedralSphere ? sv : 1.0f - sv;
    const float y = image_v * static_cast<float>(height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const auto at = [&](int px, int py) {
        if (wrap == TextureWrap2D::Repeat) {
            px %= width;
            py %= height;
            if (px < 0) px += width;
            if (py < 0) py += height;
        } else {
            px %= width;
            if (px < 0) px += width;
            py = std::clamp(py, 0, height - 1);
        }
        return pixels[static_cast<size_t>(py) * static_cast<size_t>(width) + static_cast<size_t>(px)];
    };
    const Vec3 a = lerp_vec3(at(x0, y0), at(x0 + 1, y0), tx);
    const Vec3 b = lerp_vec3(at(x0, y0 + 1), at(x0 + 1, y0 + 1), tx);
    return lerp_vec3(a, b, ty);
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

uint16_t read_u16(const std::vector<unsigned char>& data, size_t& pos) {
    if (pos + 2 > data.size()) return 0;
    const uint16_t v = static_cast<uint16_t>(data[pos]) | (static_cast<uint16_t>(data[pos + 1]) << 8);
    pos += 2;
    return v;
}

uint32_t read_u32(const std::vector<unsigned char>& data, size_t& pos) {
    if (pos + 4 > data.size()) return 0;
    const uint32_t v = static_cast<uint32_t>(data[pos]) |
        (static_cast<uint32_t>(data[pos + 1]) << 8) |
        (static_cast<uint32_t>(data[pos + 2]) << 16) |
        (static_cast<uint32_t>(data[pos + 3]) << 24);
    pos += 4;
    return v;
}

uint64_t read_u64(const std::vector<unsigned char>& data, size_t& pos) {
    const uint64_t lo = read_u32(data, pos);
    const uint64_t hi = read_u32(data, pos);
    return lo | (hi << 32);
}

std::string read_cstring(const std::vector<unsigned char>& data, size_t& pos) {
    std::string s;
    while (pos < data.size() && data[pos] != 0) {
        s.push_back(static_cast<char>(data[pos++]));
    }
    if (pos < data.size()) ++pos;
    return s;
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

class BitReader {
public:
    BitReader(const unsigned char* data, size_t size) : data_(data), size_(size) {}

    uint32_t read(int bits) {
        uint32_t value = 0;
        for (int i = 0; i < bits; ++i) {
            if (byte_ >= size_) return value;
            value |= ((data_[byte_] >> bit_) & 1u) << i;
            if (++bit_ == 8) {
                bit_ = 0;
                ++byte_;
            }
        }
        return value;
    }

    void align_byte() {
        if (bit_ != 0) {
            bit_ = 0;
            ++byte_;
        }
    }

    size_t byte_pos() const { return byte_; }
    int bit_pos() const { return bit_; }

private:
    const unsigned char* data_ = nullptr;
    size_t size_ = 0;
    size_t byte_ = 0;
    int bit_ = 0;
};

struct HuffmanTable {
    struct Entry {
        uint16_t symbol = 0;
        uint8_t length = 0;
    };
    std::vector<Entry> entries;
    int max_bits = 0;
};

uint32_t reverse_bits(uint32_t code, int length) {
    uint32_t result = 0;
    for (int i = 0; i < length; ++i) {
        result = (result << 1) | (code & 1u);
        code >>= 1;
    }
    return result;
}

bool build_huffman(const std::vector<uint8_t>& lengths, HuffmanTable& table) {
    table = {};
    for (uint8_t len : lengths) table.max_bits = std::max(table.max_bits, static_cast<int>(len));
    if (table.max_bits <= 0 || table.max_bits > 15) return false;
    table.entries.assign(size_t{1} << table.max_bits, {});
    int bl_count[16] = {};
    for (uint8_t len : lengths) {
        if (len > 15) return false;
        if (len) ++bl_count[len];
    }
    int code = 0;
    int next_code[16] = {};
    for (int bits = 1; bits <= 15; ++bits) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (int symbol = 0; symbol < static_cast<int>(lengths.size()); ++symbol) {
        const int len = lengths[static_cast<size_t>(symbol)];
        if (!len) continue;
        const uint32_t rev = reverse_bits(static_cast<uint32_t>(next_code[len]++), len);
        const uint32_t fill = 1u << (table.max_bits - len);
        for (uint32_t i = 0; i < fill; ++i) {
            const uint32_t index = rev | (i << len);
            table.entries[index] = {static_cast<uint16_t>(symbol), static_cast<uint8_t>(len)};
        }
    }
    return true;
}

int decode_symbol(BitReader& bits, const HuffmanTable& table) {
    uint32_t code = 0;
    for (int len = 1; len <= table.max_bits; ++len) {
        code |= bits.read(1) << (len - 1);
        const HuffmanTable::Entry& entry = table.entries[code];
        if (entry.length == len) return entry.symbol;
    }
    return -1;
}

bool inflate_zlib(const unsigned char* data, size_t size, std::vector<unsigned char>& out, size_t expected_size) {
    if (size < 2) return false;
    size_t pos = 0;
    const unsigned char cmf = data[pos++];
    const unsigned char flg = data[pos++];
    if ((cmf & 0x0f) != 8 || (((static_cast<int>(cmf) << 8) + flg) % 31) != 0 || (flg & 0x20)) return false;
    BitReader bits(data + pos, size - pos);
    out.clear();
    out.reserve(expected_size);
    bool final_block = false;
    static constexpr int length_base[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static constexpr int length_extra[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    static constexpr int dist_base[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    static constexpr int dist_extra[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    while (!final_block) {
        final_block = bits.read(1) != 0;
        const int type = static_cast<int>(bits.read(2));
        if (type == 0) {
            bits.align_byte();
            const size_t bpos = pos + bits.byte_pos();
            if (bpos + 4 > size) return false;
            const uint16_t len = static_cast<uint16_t>(data[bpos]) | (static_cast<uint16_t>(data[bpos + 1]) << 8);
            const uint16_t nlen = static_cast<uint16_t>(data[bpos + 2]) | (static_cast<uint16_t>(data[bpos + 3]) << 8);
            if ((len ^ 0xffffu) != nlen || bpos + 4u + len > size) return false;
            out.insert(out.end(), data + bpos + 4, data + bpos + 4 + len);
            bits = BitReader(data + bpos + 4 + len, size - (bpos + 4 + len));
            pos = bpos + 4 + len;
        } else if (type == 1 || type == 2) {
            HuffmanTable lit_table;
            HuffmanTable dist_table;
            if (type == 1) {
                std::vector<uint8_t> lit_lengths(288, 0);
                for (int i = 0; i <= 143; ++i) lit_lengths[i] = 8;
                for (int i = 144; i <= 255; ++i) lit_lengths[i] = 9;
                for (int i = 256; i <= 279; ++i) lit_lengths[i] = 7;
                for (int i = 280; i <= 287; ++i) lit_lengths[i] = 8;
                std::vector<uint8_t> dist_lengths(32, 5);
                if (!build_huffman(lit_lengths, lit_table) || !build_huffman(dist_lengths, dist_table)) return false;
            } else {
                const int hlit = static_cast<int>(bits.read(5)) + 257;
                const int hdist = static_cast<int>(bits.read(5)) + 1;
                const int hclen = static_cast<int>(bits.read(4)) + 4;
                static constexpr int order[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
                std::vector<uint8_t> code_lengths(19, 0);
                for (int i = 0; i < hclen; ++i) code_lengths[order[i]] = static_cast<uint8_t>(bits.read(3));
                HuffmanTable code_table;
                if (!build_huffman(code_lengths, code_table)) return false;
                std::vector<uint8_t> lengths;
                lengths.reserve(static_cast<size_t>(hlit + hdist));
                while (static_cast<int>(lengths.size()) < hlit + hdist) {
                    const int sym = decode_symbol(bits, code_table);
                    if (sym < 0) return false;
                    if (sym <= 15) {
                        lengths.push_back(static_cast<uint8_t>(sym));
                    } else if (sym == 16) {
                        if (lengths.empty()) return false;
                        const int repeat = static_cast<int>(bits.read(2)) + 3;
                        lengths.insert(lengths.end(), repeat, lengths.back());
                    } else if (sym == 17) {
                        lengths.insert(lengths.end(), static_cast<int>(bits.read(3)) + 3, 0);
                    } else if (sym == 18) {
                        lengths.insert(lengths.end(), static_cast<int>(bits.read(7)) + 11, 0);
                    }
                }
                std::vector<uint8_t> lit_lengths(lengths.begin(), lengths.begin() + hlit);
                std::vector<uint8_t> dist_lengths(lengths.begin() + hlit, lengths.end());
                if (!build_huffman(lit_lengths, lit_table) || !build_huffman(dist_lengths, dist_table)) return false;
            }
            while (true) {
                const int sym = decode_symbol(bits, lit_table);
                if (sym < 0) return false;
                if (sym < 256) {
                    out.push_back(static_cast<unsigned char>(sym));
                } else if (sym == 256) {
                    break;
                } else if (sym <= 285) {
                    const int li = sym - 257;
                    int len = length_base[li] + static_cast<int>(bits.read(length_extra[li]));
                    const int dsym = decode_symbol(bits, dist_table);
                    if (dsym < 0 || dsym >= 30) return false;
                    const int dist = dist_base[dsym] + static_cast<int>(bits.read(dist_extra[dsym]));
                    if (dist <= 0 || static_cast<size_t>(dist) > out.size()) return false;
                    while (len-- > 0) out.push_back(out[out.size() - static_cast<size_t>(dist)]);
                } else {
                    return false;
                }
            }
        } else {
            return false;
        }
        if (out.size() > expected_size * 2u + 1024u) return false;
    }
    return out.size() == expected_size;
}

void exr_unzip_reconstruct(std::vector<unsigned char>& bytes) {
    for (size_t i = 1; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>(static_cast<int>(bytes[i]) + static_cast<int>(bytes[i - 1]) - 128);
    }
    std::vector<unsigned char> reordered(bytes.size());
    const size_t first_count = (bytes.size() + 1u) / 2u;
    size_t even = 0;
    size_t odd = first_count;
    for (size_t i = 0; i < bytes.size(); ++i) {
        reordered[i] = (i & 1u) ? bytes[odd++] : bytes[even++];
    }
    bytes.swap(reordered);
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
    texture.alpha.assign(texture.pixels.size(), 1.0f);
    for (size_t i = 0; i < texture.pixels.size(); ++i) {
        texture.pixels[i] = {
            static_cast<float>(rgba[i * 4u + 0u]) / 255.0f,
            static_cast<float>(rgba[i * 4u + 1u]) / 255.0f,
            static_cast<float>(rgba[i * 4u + 2u]) / 255.0f,
        };
        texture.alpha[i] = static_cast<float>(rgba[i * 4u + 3u]) / 255.0f;
    }
    texture.build_mips();
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
    return sample_level(pixels, width, height, uv);
}

Vec3 Texture::sample_lod(Vec2 uv, float lod) const {
    return sample_lod(uv, lod, TextureWrap2D::Repeat);
}

Vec3 Texture::sample_lod(Vec2 uv, float lod, TextureWrap2D wrap) const {
    if (width <= 0 || height <= 0 || pixels.empty()) {
        return {1.0f, 1.0f, 1.0f};
    }
    if (mip_pixels.empty() || mip_widths.empty() || mip_heights.empty()) {
        return sample_level(pixels, width, height, uv, wrap);
    }
    lod = std::max(0.0f, lod);
    const int last = static_cast<int>(mip_pixels.size()) - 1;
    const int l0 = std::clamp(static_cast<int>(std::floor(lod)), 0, last);
    const int l1 = std::min(l0 + 1, last);
    const float t = std::clamp(lod - static_cast<float>(l0), 0.0f, 1.0f);
    const Vec3 a = sample_level(mip_pixels[static_cast<size_t>(l0)], mip_widths[static_cast<size_t>(l0)], mip_heights[static_cast<size_t>(l0)], uv, wrap);
    const Vec3 b = sample_level(mip_pixels[static_cast<size_t>(l1)], mip_widths[static_cast<size_t>(l1)], mip_heights[static_cast<size_t>(l1)], uv, wrap);
    return lerp_vec3(a, b, t);
}

float Texture::sample_alpha(Vec2 uv) const {
    if (width <= 0 || height <= 0 || alpha.empty()) {
        return 1.0f;
    }
    const float u = wrap01(uv.x);
    const float v = wrap01(uv.y);
    const int x = std::clamp(static_cast<int>(u * static_cast<float>(width)), 0, width - 1);
    const int y = std::clamp(static_cast<int>((1.0f - v) * static_cast<float>(height)), 0, height - 1);
    return alpha[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
}

void Texture::build_mips() {
    mip_widths.clear();
    mip_heights.clear();
    mip_pixels.clear();
    if (width <= 0 || height <= 0 || pixels.empty()) {
        return;
    }
    mip_widths.push_back(width);
    mip_heights.push_back(height);
    mip_pixels.push_back(pixels);
    int current_width = width;
    int current_height = height;
    while (current_width > 1 || current_height > 1) {
        const int next_width = std::max(1, current_width / 2);
        const int next_height = std::max(1, current_height / 2);
        const std::vector<Vec3>& source = mip_pixels.back();
        std::vector<Vec3> next(static_cast<size_t>(next_width) * static_cast<size_t>(next_height));
        const auto at = [&](int x, int y) {
            x = std::min(x, current_width - 1);
            y = std::min(y, current_height - 1);
            return source[static_cast<size_t>(y) * static_cast<size_t>(current_width) + static_cast<size_t>(x)];
        };
        for (int y = 0; y < next_height; ++y) {
            for (int x = 0; x < next_width; ++x) {
                const int sx = x * 2;
                const int sy = y * 2;
                next[static_cast<size_t>(y) * static_cast<size_t>(next_width) + static_cast<size_t>(x)] =
                    (at(sx, sy) + at(sx + 1, sy) + at(sx, sy + 1) + at(sx + 1, sy + 1)) * 0.25f;
            }
        }
        mip_widths.push_back(next_width);
        mip_heights.push_back(next_height);
        mip_pixels.push_back(std::move(next));
        current_width = next_width;
        current_height = next_height;
    }
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
    texture.alpha.assign(texture.pixels.size(), 1.0f);
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
    texture.build_mips();
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
    texture.alpha.assign(texture.pixels.size(), 1.0f);
    for (int y = 0; y < height; ++y) {
        if (!read_hdr_scanline(in, width, texture.pixels, y)) {
            error = "HDR texture ended early: " + path;
            return false;
        }
    }
    texture.build_mips();
    return true;
}

bool load_exr_texture(const std::string& name, const std::string& path, Texture& texture, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Could not open EXR texture: " + path;
        return false;
    }
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() < 16) {
        error = "Invalid EXR texture: " + path;
        return false;
    }
    size_t pos = 0;
    const uint32_t magic = read_u32(data, pos);
    const uint32_t version = read_u32(data, pos);
    if (magic != 20000630u || (version & 0xffu) != 2u) {
        error = "Invalid EXR texture: " + path;
        return false;
    }

    struct Channel {
        std::string name;
        int pixel_type = -1;
    };
    std::vector<Channel> channels;
    int compression = -1;
    int min_x = 0;
    int min_y = 0;
    int max_x = -1;
    int max_y = -1;

    while (pos < data.size()) {
        const std::string attr_name = read_cstring(data, pos);
        if (attr_name.empty()) break;
        const std::string attr_type = read_cstring(data, pos);
        const uint32_t attr_size = read_u32(data, pos);
        if (pos + attr_size > data.size()) {
            error = "Invalid EXR attribute data: " + path;
            return false;
        }
        const size_t attr_pos = pos;
        if (attr_name == "channels" && attr_type == "chlist") {
            size_t p = attr_pos;
            while (p < attr_pos + attr_size && data[p] != 0) {
                Channel channel;
                channel.name = read_cstring(data, p);
                channel.pixel_type = static_cast<int>(read_u32(data, p));
                p += 12; // pLinear/reserved + xSampling + ySampling
                channels.push_back(std::move(channel));
            }
        } else if (attr_name == "compression" && attr_type == "compression" && attr_size == 1) {
            compression = data[attr_pos];
        } else if (attr_name == "dataWindow" && attr_type == "box2i" && attr_size == 16) {
            size_t p = attr_pos;
            min_x = static_cast<int>(read_u32(data, p));
            min_y = static_cast<int>(read_u32(data, p));
            max_x = static_cast<int>(read_u32(data, p));
            max_y = static_cast<int>(read_u32(data, p));
        }
        pos += attr_size;
    }

    const int width = max_x - min_x + 1;
    const int height = max_y - min_y + 1;
    if (width <= 0 || height <= 0 || channels.empty()) {
        error = "Invalid EXR dimensions/channels: " + path;
        return false;
    }
    if (compression != 0 && compression != 3) {
        error = "Unsupported EXR compression (only none/ZIP): " + path;
        return false;
    }
    for (const Channel& channel : channels) {
        if (channel.pixel_type != 1 && channel.pixel_type != 2) {
            error = "Unsupported EXR channel type: " + path;
            return false;
        }
    }

    const int lines_per_chunk = compression == 3 ? 16 : 1;
    const int chunk_count = (height + lines_per_chunk - 1) / lines_per_chunk;
    if (pos + static_cast<size_t>(chunk_count) * 8u > data.size()) {
        error = "Invalid EXR chunk table: " + path;
        return false;
    }
    std::vector<uint64_t> offsets(static_cast<size_t>(chunk_count));
    for (uint64_t& offset : offsets) offset = read_u64(data, pos);

    texture = {};
    texture.name = name;
    texture.path = path;
    texture.width = width;
    texture.height = height;
    texture.pixels.assign(static_cast<size_t>(width) * static_cast<size_t>(height), {});
    texture.alpha.assign(texture.pixels.size(), 1.0f);

    auto channel_index = [&](const char* wanted) {
        for (int i = 0; i < static_cast<int>(channels.size()); ++i) {
            if (channels[static_cast<size_t>(i)].name == wanted) return i;
        }
        return -1;
    };
    const int r_index = channel_index("R");
    const int g_index = channel_index("G");
    const int b_index = channel_index("B");
    if (r_index < 0 || g_index < 0 || b_index < 0) {
        error = "EXR texture must contain R/G/B channels: " + path;
        return false;
    }

    const auto bytes_per_channel = [&](const Channel& channel) {
        return channel.pixel_type == 1 ? 2 : 4;
    };
    int bytes_per_pixel = 0;
    for (const Channel& channel : channels) bytes_per_pixel += bytes_per_channel(channel);

    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        size_t chunk_pos = static_cast<size_t>(offsets[static_cast<size_t>(chunk)]);
        if (chunk_pos + 8 > data.size()) {
            error = "Invalid EXR chunk offset: " + path;
            return false;
        }
        const int y = static_cast<int>(read_u32(data, chunk_pos));
        const uint32_t packed_size = read_u32(data, chunk_pos);
        if (chunk_pos + packed_size > data.size()) {
            error = "Invalid EXR chunk size: " + path;
            return false;
        }
        const int line_count = std::min(lines_per_chunk, max_y - y + 1);
        const size_t unpacked_size = static_cast<size_t>(line_count) * static_cast<size_t>(width) * static_cast<size_t>(bytes_per_pixel);
        std::vector<unsigned char> raw;
        if (compression == 0) {
            if (packed_size != unpacked_size) {
                error = "Invalid uncompressed EXR chunk: " + path;
                return false;
            }
            raw.assign(data.begin() + static_cast<std::ptrdiff_t>(chunk_pos), data.begin() + static_cast<std::ptrdiff_t>(chunk_pos + packed_size));
        } else {
            if (!inflate_zlib(data.data() + chunk_pos, packed_size, raw, unpacked_size)) {
                error = "Could not decompress EXR ZIP chunk: " + path;
                return false;
            }
            exr_unzip_reconstruct(raw);
        }

        size_t raw_pos = 0;
        for (int ly = 0; ly < line_count; ++ly) {
            const int out_y = y - min_y + ly;
            for (int c = 0; c < static_cast<int>(channels.size()); ++c) {
                const int bpc = bytes_per_channel(channels[static_cast<size_t>(c)]);
                for (int x = 0; x < width; ++x) {
                    float value = 0.0f;
                    if (bpc == 2) {
                        size_t p = raw_pos;
                        value = half_to_float(read_u16(raw, p));
                    } else {
                        uint32_t bits = static_cast<uint32_t>(raw[raw_pos]) |
                            (static_cast<uint32_t>(raw[raw_pos + 1]) << 8) |
                            (static_cast<uint32_t>(raw[raw_pos + 2]) << 16) |
                            (static_cast<uint32_t>(raw[raw_pos + 3]) << 24);
                        std::memcpy(&value, &bits, sizeof(value));
                    }
                    raw_pos += static_cast<size_t>(bpc);
                    Vec3& pixel = texture.pixels[static_cast<size_t>(out_y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
                    if (c == r_index) pixel.x = value;
                    else if (c == g_index) pixel.y = value;
                    else if (c == b_index) pixel.z = value;
                }
            }
        }
    }
    texture.build_mips();
    return true;
}

bool load_texture_file(const std::string& name, const std::string& path, Texture& texture, std::string& error) {
    const std::string ext = lowercase_extension(path);
    if (ext == ".exr") {
        return load_exr_texture(name, path, texture, error);
    }
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

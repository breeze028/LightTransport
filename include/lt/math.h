#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace lt {

#if defined(__CUDACC__)
#define LT_HOST_DEVICE __host__ __device__
#else
#define LT_HOST_DEVICE
#endif

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInfinity = std::numeric_limits<float>::infinity();

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    LT_HOST_DEVICE constexpr Vec3() {}
    LT_HOST_DEVICE constexpr Vec3(float s) : x(s), y(s), z(s) {}
    LT_HOST_DEVICE constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    LT_HOST_DEVICE constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    LT_HOST_DEVICE constexpr Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    LT_HOST_DEVICE constexpr Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    LT_HOST_DEVICE constexpr Vec3 operator*(Vec3 b) const { return {x * b.x, y * b.y, z * b.z}; }
    LT_HOST_DEVICE constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    LT_HOST_DEVICE constexpr Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }

    LT_HOST_DEVICE Vec3& operator+=(Vec3 b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }

    LT_HOST_DEVICE Vec3& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
};

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    LT_HOST_DEVICE constexpr Vec2() {}
    LT_HOST_DEVICE constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}
    LT_HOST_DEVICE constexpr Vec2 operator+(Vec2 b) const { return {x + b.x, y + b.y}; }
    LT_HOST_DEVICE constexpr Vec2 operator-(Vec2 b) const { return {x - b.x, y - b.y}; }
    LT_HOST_DEVICE constexpr Vec2 operator*(float s) const { return {x * s, y * s}; }
};

LT_HOST_DEVICE inline constexpr Vec3 operator*(float s, Vec3 v) { return v * s; }

LT_HOST_DEVICE inline constexpr float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

LT_HOST_DEVICE inline constexpr Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline float length(Vec3 v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(Vec3 v) {
    const float len = length(v);
    return len > 0.0f ? v / len : Vec3{};
}

inline Vec3 clamp(Vec3 v, float lo = 0.0f, float hi = 1.0f) {
    return {
        std::isfinite(v.x) ? std::clamp(v.x, lo, hi) : 0.0f,
        std::isfinite(v.y) ? std::clamp(v.y, lo, hi) : 0.0f,
        std::isfinite(v.z) ? std::clamp(v.z, lo, hi) : 0.0f,
    };
}

inline Vec3 min(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

inline Vec3 max(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

inline Vec3 face_forward(Vec3 n, Vec3 ray_dir) {
    return dot(n, ray_dir) < 0.0f ? n : -n;
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Rng {
    uint32_t state = 1;

    explicit Rng(uint32_t seed = 1) : state(seed ? seed : 1) {}

    uint32_t next_u32() {
        state = state * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    float next_float() {
        return static_cast<float>((next_u32() >> 8) * (1.0 / 16777216.0));
    }
};

LT_HOST_DEVICE inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

LT_HOST_DEVICE inline uint32_t make_pixel_seed(uint32_t x, uint32_t y, uint32_t frame) {
    return hash_u32(x * 0x8da6b343u ^ y * 0xd8163841u ^ frame * 0xcb1ab31fu ^ 0x9e3779b9u);
}

inline Vec3 cosine_sample_hemisphere(float u1, float u2) {
    const float r = std::sqrt(u1);
    const float phi = 2.0f * kPi * u2;
    return {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1.0f - u1))};
}

inline Vec3 to_world(Vec3 local, Vec3 n) {
    const Vec3 up = std::fabs(n.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(up, n));
    const Vec3 bitangent = cross(n, tangent);
    return tangent * local.x + bitangent * local.y + n * local.z;
}

inline uint32_t to_rgba8(Vec3 color) {
    color = clamp(color);
    const auto encode = [](float v) {
        return static_cast<uint32_t>(std::clamp(std::pow(v, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    const uint32_t r = encode(color.x);
    const uint32_t g = encode(color.y);
    const uint32_t b = encode(color.z);
    return 0xff000000u | (r << 16u) | (g << 8u) | b;
}

} // namespace lt

#undef LT_HOST_DEVICE

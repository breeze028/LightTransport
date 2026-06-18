__host__ __device__ Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__host__ __device__ Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ Vec3 mul(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
__host__ __device__ Vec3 mul(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__host__ __device__ Vec2 add(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
__host__ __device__ Vec2 mul(Vec2 a, float s) { return {a.x * s, a.y * s}; }
__host__ __device__ Vec3 divv(Vec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
__host__ __device__ float ddot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ Vec3 dcross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ Vec3 dnormalize(Vec3 v) {
    const float len = sqrtf(ddot(v, v));
    return len > 0.0f ? divv(v, len) : Vec3{};
}
__host__ __device__ float dclamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}
__host__ __device__ float wrap01_gpu(float v) {
    v = v - floorf(v);
    return v < 0.0f ? v + 1.0f : v;
}
__host__ __device__ int iclamp_gpu(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__host__ __device__ Vec3 dlerp(Vec3 a, Vec3 b, float t) {
    return add(mul(a, 1.0f - t), mul(b, t));
}
__host__ __device__ uint32_t rng_next(uint32_t& state) {
    state = state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
__host__ __device__ float rng_float(uint32_t& state) {
    return static_cast<float>((rng_next(state) >> 8) * (1.0 / 16777216.0));
}
__device__ Vec3 cosine_sample(float u1, float u2) {
    const float r = sqrtf(u1);
    const float phi = 2.0f * kPi * u2;
    return {r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1))};
}
__device__ Vec3 to_world_gpu(Vec3 local, Vec3 n) {
    const Vec3 up = fabsf(n.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = dnormalize(dcross(up, n));
    const Vec3 bitangent = dcross(n, tangent);
    return add(add(mul(tangent, local.x), mul(bitangent, local.y)), mul(n, local.z));
}
__device__ uint32_t rgba8_gpu(Vec3 color) {
    color.x = isfinite(color.x) ? dclamp(color.x, 0.0f, 1.0f) : 0.0f;
    color.y = isfinite(color.y) ? dclamp(color.y, 0.0f, 1.0f) : 0.0f;
    color.z = isfinite(color.z) ? dclamp(color.z, 0.0f, 1.0f) : 0.0f;
    const uint32_t r = static_cast<uint32_t>(dclamp(powf(color.x, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f + 0.5f);
    const uint32_t g = static_cast<uint32_t>(dclamp(powf(color.y, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f + 0.5f);
    const uint32_t b = static_cast<uint32_t>(dclamp(powf(color.z, 1.0f / 2.2f), 0.0f, 1.0f) * 255.0f + 0.5f);
    return 0xff000000u | (r << 16u) | (g << 8u) | b;
}

__device__ Ray camera_ray(const Camera& camera, int x, int y, int width, int height, uint32_t& rng) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float half_height = tanf(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = dnormalize(sub(camera.target, camera.position));
    const float right_sign = camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const Vec3 right = mul(dnormalize(dcross(forward, camera.up)), right_sign);
    const Vec3 up = mul(dcross(right, forward), right_sign);
    const float u = ((static_cast<float>(x) + rng_float(rng)) / static_cast<float>(width) * 2.0f - 1.0f) * half_width;
    const float v = (1.0f - (static_cast<float>(y) + rng_float(rng)) / static_cast<float>(height) * 2.0f) * half_height;
    return {camera.position, dnormalize(add(add(forward, mul(right, u)), mul(up, v)))};
}


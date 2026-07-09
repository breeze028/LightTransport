#include "lt/denoiser.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lt {
namespace {

float luminance(Vec3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

Vec3 safe_demodulate(Vec3 radiance, Vec3 emission, Vec3 albedo) {
    const Vec3 lighting = max(radiance - emission, Vec3{});
    return {
        lighting.x / std::max(albedo.x, 0.001f),
        lighting.y / std::max(albedo.y, 0.001f),
        lighting.z / std::max(albedo.z, 0.001f),
    };
}

Vec3 lerp(Vec3 a, Vec3 b, float t) {
    return a * (1.0f - t) + b * t;
}

float lerp(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

bool has_valid_surface(const Framebuffer::AovBuffers& aov, size_t index) {
    return aov.object_id[index] != 0u &&
        std::isfinite(aov.linear_depth[index]) &&
        aov.linear_depth[index] > 0.0f &&
        dot(aov.normal[index], aov.normal[index]) > 0.0f;
}

bool project_to_pixel(const Camera& camera,
                      Vec3 world_position,
                      int width,
                      int height,
                      float jitter_x,
                      float jitter_y,
                      float& pixel_x,
                      float& pixel_y,
                      float& linear_depth) {
    const float aspect = static_cast<float>(width) / static_cast<float>(std::max(1, height));
    const float half_height = std::tan(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = normalize(camera.target - camera.position);
    const float right_sign = camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const Vec3 right = normalize(cross(forward, camera.up)) * right_sign;
    const Vec3 up = cross(right, forward) * right_sign;
    const Vec3 rel = world_position - camera.position;
    linear_depth = dot(rel, forward);
    if (!std::isfinite(linear_depth) || linear_depth <= 0.0001f) {
        return false;
    }
    const float ndc_x = dot(rel, right) / (linear_depth * half_width);
    const float ndc_y = dot(rel, up) / (linear_depth * half_height);
    if (!std::isfinite(ndc_x) || !std::isfinite(ndc_y) ||
        ndc_x < -1.25f || ndc_x > 1.25f || ndc_y < -1.25f || ndc_y > 1.25f) {
        return false;
    }
    pixel_x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(width) - jitter_x;
    pixel_y = (0.5f - ndc_y * 0.5f) * static_cast<float>(height) - jitter_y;
    return pixel_x >= 0.0f && pixel_y >= 0.0f &&
        pixel_x < static_cast<float>(width) && pixel_y < static_cast<float>(height);
}

bool invalidates_svgf_history(RenderDirty dirty) {
    return has_dirty(dirty, RenderDirty::Geometry) ||
        has_dirty(dirty, RenderDirty::Transform) ||
        has_dirty(dirty, RenderDirty::Material) ||
        has_dirty(dirty, RenderDirty::Texture) ||
        has_dirty(dirty, RenderDirty::Environment) ||
        has_dirty(dirty, RenderDirty::IrradianceVolume) ||
        has_dirty(dirty, RenderDirty::Lightmap);
}

bool invalidates_taa_history(RenderDirty dirty) {
    return has_dirty(dirty, RenderDirty::Geometry) ||
        has_dirty(dirty, RenderDirty::Material) ||
        has_dirty(dirty, RenderDirty::Texture) ||
        has_dirty(dirty, RenderDirty::Environment) ||
        has_dirty(dirty, RenderDirty::IrradianceVolume) ||
        has_dirty(dirty, RenderDirty::Lightmap);
}

bool reproject_history(const Framebuffer& framebuffer,
                       const RenderSettings& settings,
                       size_t index,
                       Vec3& prev_illumination,
                       float& prev_moment1,
                       float& prev_moment2,
                       float& prev_history) {
    const Framebuffer::AovBuffers& aov = framebuffer.aov;
    const Framebuffer::SvgfHistory& history = framebuffer.svgf_history;
    if (!history.valid || !has_valid_surface(aov, index)) {
        return false;
    }

    float prev_x = 0.0f;
    float prev_y = 0.0f;
    float prev_linear_depth = 0.0f;
    if (!project_to_pixel(
            history.camera,
            aov.world_position[index],
            settings.width,
            settings.height,
            history.jitter_x,
            history.jitter_y,
            prev_x,
            prev_y,
            prev_linear_depth)) {
        return false;
    }

    const int px = std::clamp(static_cast<int>(std::floor(prev_x)), 0, settings.width - 1);
    const int py = std::clamp(static_cast<int>(std::floor(prev_y)), 0, settings.height - 1);
    const size_t prev_index = static_cast<size_t>(py) * static_cast<size_t>(settings.width) + static_cast<size_t>(px);
    if (prev_index >= history.illumination.size() || history.history_length[prev_index] <= 0.0f) {
        return false;
    }
    if (history.object_id[prev_index] != aov.object_id[index]) {
        return false;
    }

    const float depth_tolerance = std::max(0.02f, 0.06f * std::max(aov.linear_depth[index], 1.0f));
    if (std::abs(history.linear_depth[prev_index] - prev_linear_depth) > depth_tolerance) {
        return false;
    }
    if (dot(history.normal[prev_index], aov.normal[index]) < 0.72f) {
        return false;
    }

    prev_illumination = history.illumination[prev_index];
    prev_moment1 = history.moment1[prev_index];
    prev_moment2 = history.moment2[prev_index];
    prev_history = history.history_length[prev_index];
    return true;
}

float edge_stopping_weight(const Framebuffer::AovBuffers& aov,
                           const std::vector<Vec3>& illumination,
                           const std::vector<float>& variance,
                           int width,
                           int height,
                           int center_x,
                           int center_y,
                           int sample_x,
                           int sample_y,
                           const RenderSettings& settings) {
    if (sample_x < 0 || sample_y < 0 || sample_x >= width || sample_y >= height) {
        return 0.0f;
    }
    const size_t center = static_cast<size_t>(center_y) * static_cast<size_t>(width) + static_cast<size_t>(center_x);
    const size_t sample = static_cast<size_t>(sample_y) * static_cast<size_t>(width) + static_cast<size_t>(sample_x);
    if (aov.object_id[center] != aov.object_id[sample]) {
        return 0.0f;
    }
    if (!has_valid_surface(aov, center) || !has_valid_surface(aov, sample)) {
        return center == sample ? 1.0f : 0.0f;
    }

    const float normal_weight = std::exp(-std::max(0.0f, 1.0f - dot(aov.normal[center], aov.normal[sample])) * settings.svgf_phi_normal);
    const float depth_scale = std::max(0.01f, settings.svgf_phi_depth * 0.015f * std::max(aov.linear_depth[center], 1.0f));
    const float depth_weight = std::exp(-std::abs(aov.linear_depth[center] - aov.linear_depth[sample]) / depth_scale);
    const float color_scale = std::max(0.01f, settings.svgf_phi_color * std::sqrt(std::max(variance[center], 0.0f)) + 0.001f);
    const float color_weight = std::exp(-std::abs(luminance(illumination[center]) - luminance(illumination[sample])) / color_scale);
    return normal_weight * depth_weight * color_weight;
}

void atrous_pass(const Framebuffer::AovBuffers& aov,
                 const std::vector<Vec3>& input_illumination,
                 const std::vector<float>& input_variance,
                 int width,
                 int height,
                 int step,
                 const RenderSettings& settings,
                 std::vector<Vec3>& output_illumination,
                 std::vector<float>& output_variance) {
    static constexpr float kernel[5] = {1.0f / 16.0f, 1.0f / 4.0f, 3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            if (!has_valid_surface(aov, index)) {
                output_illumination[index] = input_illumination[index];
                output_variance[index] = input_variance[index];
                continue;
            }

            Vec3 sum{};
            float variance_sum = 0.0f;
            float weight_sum = 0.0f;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    const int sx = x + kx * step;
                    const int sy = y + ky * step;
                    const float kernel_weight = kernel[kx + 2] * kernel[ky + 2];
                    const float edge_weight = edge_stopping_weight(
                        aov, input_illumination, input_variance, width, height, x, y, sx, sy, settings);
                    const float weight = kernel_weight * edge_weight;
                    if (weight <= 0.0f) {
                        continue;
                    }
                    const size_t sample = static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx);
                    sum += input_illumination[sample] * weight;
                    variance_sum += input_variance[sample] * weight;
                    weight_sum += weight;
                }
            }

            if (weight_sum > 0.0f) {
                output_illumination[index] = sum / weight_sum;
                output_variance[index] = variance_sum / weight_sum;
            } else {
                output_illumination[index] = input_illumination[index];
                output_variance[index] = input_variance[index];
            }
        }
    }
}

Vec3 debug_color(const Framebuffer::AovBuffers& aov,
                 const std::vector<Vec3>& filtered_illumination,
                 const std::vector<float>& variance,
                 const std::vector<float>& history_length,
                 size_t index,
                 const RenderSettings& settings) {
    switch (settings.svgf_debug_view) {
    case DenoiserDebugView::Raw:
        return aov.radiance[index];
    case DenoiserDebugView::Albedo:
        return aov.albedo[index];
    case DenoiserDebugView::Normal:
        return aov.normal[index] * 0.5f + Vec3{0.5f, 0.5f, 0.5f};
    case DenoiserDebugView::Depth: {
        const float depth = std::isfinite(aov.linear_depth[index]) ? aov.linear_depth[index] : 0.0f;
        const float normalized = depth / (depth + 10.0f);
        return {normalized, normalized, normalized};
    }
    case DenoiserDebugView::Variance: {
        const float v = std::clamp(std::sqrt(std::max(0.0f, variance[index])) * 0.25f, 0.0f, 1.0f);
        return {v, v, v};
    }
    case DenoiserDebugView::HistoryLength: {
        const float h = std::clamp(history_length[index] / 32.0f, 0.0f, 1.0f);
        return {h, h, h};
    }
    case DenoiserDebugView::Final:
    default:
        return filtered_illumination[index] * aov.albedo[index] + aov.emission[index];
    }
}

Vec3 min_vec(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

Vec3 max_vec(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

Vec3 clamp_vec(Vec3 v, Vec3 lo, Vec3 hi) {
    return {
        std::clamp(v.x, lo.x, hi.x),
        std::clamp(v.y, lo.y, hi.y),
        std::clamp(v.z, lo.z, hi.z),
    };
}

Vec3 rgb_to_ycocg(Vec3 c) {
    const float co = c.x - c.z;
    const float t = c.z + co * 0.5f;
    const float cg = c.y - t;
    const float y = t + cg * 0.5f;
    return {y, co, cg};
}

Vec3 ycocg_to_rgb(Vec3 c) {
    const float t = c.x - c.z * 0.5f;
    const float g = c.z + t;
    const float b = t - c.y * 0.5f;
    const float r = b + c.y;
    return {r, g, b};
}

void neighborhood_ycocg_stats(const std::vector<Vec3>& color,
                              int width,
                              int height,
                              int x,
                              int y,
                              Vec3& lo,
                              Vec3& hi,
                              Vec3& mean,
                              Vec3& sigma) {
    lo = {kInfinity, kInfinity, kInfinity};
    hi = {-kInfinity, -kInfinity, -kInfinity};
    mean = {};
    Vec3 sum2{};
    float count = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        const int sy = std::clamp(y + dy, 0, height - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            const int sx = std::clamp(x + dx, 0, width - 1);
            const Vec3 c = rgb_to_ycocg(color[static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx)]);
            lo = min_vec(lo, c);
            hi = max_vec(hi, c);
            mean += c;
            sum2 += c * c;
            count += 1.0f;
        }
    }
    mean = mean / count;
    const Vec3 variance = max(sum2 / count - mean * mean, Vec3{});
    sigma = {std::sqrt(variance.x), std::sqrt(variance.y), std::sqrt(variance.z)};
}

Vec3 clip_history_ycocg(const std::vector<Vec3>& current_color,
                        int width,
                        int height,
                        int x,
                        int y,
                        Vec3 history_color) {
    Vec3 lo;
    Vec3 hi;
    Vec3 mean;
    Vec3 sigma;
    neighborhood_ycocg_stats(current_color, width, height, x, y, lo, hi, mean, sigma);
    const Vec3 variance_lo = mean - sigma * 1.25f;
    const Vec3 variance_hi = mean + sigma * 1.25f;
    lo = max_vec(lo, variance_lo);
    hi = min_vec(hi, variance_hi);
    return ycocg_to_rgb(clamp_vec(rgb_to_ycocg(history_color), lo, hi));
}

bool valid_taa_history_sample(const Framebuffer::AovBuffers& aov,
                              const Framebuffer::TaaHistory& history,
                              size_t current_index,
                              size_t previous_index,
                              bool surface,
                              float previous_linear_depth,
                              bool allow_object_mismatch) {
    if (previous_index >= history.color.size()) {
        return false;
    }
    const bool same_object = history.object_id[previous_index] == aov.object_id[current_index];
    if (!same_object) {
        return allow_object_mismatch;
    }
    if (!surface) {
        return true;
    }
    const float depth_tolerance = std::max(0.02f, 0.06f * std::max(aov.linear_depth[current_index], 1.0f));
    return std::abs(history.linear_depth[previous_index] - previous_linear_depth) <= depth_tolerance &&
        dot(history.normal[previous_index], aov.normal[current_index]) >= 0.72f;
}

bool taa_object_edge(const Framebuffer::AovBuffers& aov, int width, int height, int x, int y) {
    const auto at = [&](int px, int py) -> size_t {
        return static_cast<size_t>(std::clamp(py, 0, height - 1)) * static_cast<size_t>(width) +
            static_cast<size_t>(std::clamp(px, 0, width - 1));
    };
    const size_t center = at(x, y);
    return aov.object_id[center] != aov.object_id[at(x - 1, y)] ||
        aov.object_id[center] != aov.object_id[at(x + 1, y)] ||
        aov.object_id[center] != aov.object_id[at(x, y - 1)] ||
        aov.object_id[center] != aov.object_id[at(x, y + 1)];
}

bool sample_taa_history_bilinear(const Framebuffer& framebuffer,
                                 const RenderSettings& settings,
                                 size_t current_index,
                                 bool surface,
                                 float previous_x,
                                 float previous_y,
                                 float previous_linear_depth,
                                 bool allow_object_mismatch,
                                 Vec3& previous_color,
                                 float& previous_history_length) {
    const Framebuffer::AovBuffers& aov = framebuffer.aov;
    const Framebuffer::TaaHistory& history = framebuffer.taa_history;
    const int x0 = std::clamp(static_cast<int>(std::floor(previous_x)), 0, settings.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(previous_y)), 0, settings.height - 1);
    const int x1 = std::min(x0 + 1, settings.width - 1);
    const int y1 = std::min(y0 + 1, settings.height - 1);
    const float tx = std::clamp(previous_x - static_cast<float>(x0), 0.0f, 1.0f);
    const float ty = std::clamp(previous_y - static_cast<float>(y0), 0.0f, 1.0f);
    const int xs[2] = {x0, x1};
    const int ys[2] = {y0, y1};
    const float wx[2] = {1.0f - tx, tx};
    const float wy[2] = {1.0f - ty, ty};

    Vec3 sum{};
    float length_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int yy = 0; yy < 2; ++yy) {
        for (int xx = 0; xx < 2; ++xx) {
            const float w = wx[xx] * wy[yy];
            const size_t previous_index =
                static_cast<size_t>(ys[yy]) * static_cast<size_t>(settings.width) + static_cast<size_t>(xs[xx]);
            if (w <= 0.0f ||
                !valid_taa_history_sample(aov, history, current_index, previous_index, surface, previous_linear_depth, allow_object_mismatch)) {
                continue;
            }
            sum += history.color[previous_index] * w;
            length_sum += history.history_length[previous_index] * w;
            weight_sum += w;
        }
    }
    if (weight_sum <= 0.0f) {
        return false;
    }
    previous_color = sum / weight_sum;
    previous_history_length = length_sum / weight_sum;
    return true;
}

bool reproject_taa_history(const Framebuffer& framebuffer,
                           const RenderSettings& settings,
                           size_t index,
                           Vec3& previous_color,
                           float& previous_history_length) {
    const Framebuffer::AovBuffers& aov = framebuffer.aov;
    const Framebuffer::TaaHistory& history = framebuffer.taa_history;
    if (!history.valid || history.color.empty()) {
        return false;
    }

    const bool surface = has_valid_surface(aov, index);
    if (!surface && aov.object_id[index] != 0u) {
        return false;
    }

    float prev_x = 0.0f;
    float prev_y = 0.0f;
    float prev_linear_depth = 0.0f;
    const int x = static_cast<int>(index % static_cast<size_t>(settings.width));
    const int y = static_cast<int>(index / static_cast<size_t>(settings.width));
    const bool allow_object_mismatch = temporal_jitter_enabled(settings) &&
        taa_object_edge(aov, settings.width, settings.height, x, y);
    if (surface) {
        if (!project_to_pixel(
                history.camera,
                aov.world_position[index],
                settings.width,
                settings.height,
                history.jitter_x,
                history.jitter_y,
                prev_x,
                prev_y,
                prev_linear_depth)) {
            return false;
        }
    } else {
        prev_x = static_cast<float>(x) + settings.camera_jitter_x - history.jitter_x;
        prev_y = static_cast<float>(y) + settings.camera_jitter_y - history.jitter_y;
        if (prev_x < 0.0f || prev_y < 0.0f || prev_x >= static_cast<float>(settings.width) || prev_y >= static_cast<float>(settings.height)) {
            return false;
        }
    }

    return sample_taa_history_bilinear(
        framebuffer,
        settings,
        index,
        surface,
        prev_x,
        prev_y,
        prev_linear_depth,
        allow_object_mismatch,
        previous_color,
        previous_history_length);
}

void apply_luma_edge_post_aa(const Framebuffer::AovBuffers& aov,
                             int width,
                             int height,
                             const std::vector<Vec3>& input,
                             std::vector<Vec3>& output) {
    output = input;
    const auto at = [&](int x, int y) -> size_t {
        return static_cast<size_t>(std::clamp(y, 0, height - 1)) * static_cast<size_t>(width) +
            static_cast<size_t>(std::clamp(x, 0, width - 1));
    };
    const auto sample = [&](float x, float y) -> Vec3 {
        const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, width - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, height - 1);
        const int x1 = std::min(x0 + 1, width - 1);
        const int y1 = std::min(y0 + 1, height - 1);
        const float tx = std::clamp(x - static_cast<float>(x0), 0.0f, 1.0f);
        const float ty = std::clamp(y - static_cast<float>(y0), 0.0f, 1.0f);
        const Vec3 a = lerp(input[at(x0, y0)], input[at(x1, y0)], tx);
        const Vec3 b = lerp(input[at(x0, y1)], input[at(x1, y1)], tx);
        return lerp(a, b, ty);
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t center_i = at(x, y);
            const size_t nw_i = at(x - 1, y - 1);
            const size_t n_i = at(x, y - 1);
            const size_t ne_i = at(x + 1, y - 1);
            const size_t w_i = at(x - 1, y);
            const size_t e_i = at(x + 1, y);
            const size_t sw_i = at(x - 1, y + 1);
            const size_t s_i = at(x, y + 1);
            const size_t se_i = at(x + 1, y + 1);
            const float luma_m = luminance(input[center_i]);
            const float luma_nw = luminance(input[nw_i]);
            const float luma_n = luminance(input[n_i]);
            const float luma_ne = luminance(input[ne_i]);
            const float luma_w = luminance(input[w_i]);
            const float luma_e = luminance(input[e_i]);
            const float luma_sw = luminance(input[sw_i]);
            const float luma_s = luminance(input[s_i]);
            const float luma_se = luminance(input[se_i]);
            const float lmin = std::min({luma_m, luma_nw, luma_n, luma_ne, luma_w, luma_e, luma_sw, luma_s, luma_se});
            const float lmax = std::max({luma_m, luma_nw, luma_n, luma_ne, luma_w, luma_e, luma_sw, luma_s, luma_se});
            const bool geometry_edge =
                aov.object_id[center_i] != aov.object_id[w_i] ||
                aov.object_id[center_i] != aov.object_id[e_i] ||
                aov.object_id[center_i] != aov.object_id[n_i] ||
                aov.object_id[center_i] != aov.object_id[s_i];
            const float range = lmax - lmin;
            if (!geometry_edge && range < 0.025f) {
                continue;
            }
            float dir_x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
            float dir_y = ((luma_nw + luma_sw) - (luma_ne + luma_se));
            const float dir_reduce = std::max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125f, 0.0078125f);
            const float inv_min = 1.0f / (std::min(std::abs(dir_x), std::abs(dir_y)) + dir_reduce);
            dir_x = std::clamp(dir_x * inv_min, -8.0f, 8.0f);
            dir_y = std::clamp(dir_y * inv_min, -8.0f, 8.0f);
            const float px = static_cast<float>(x);
            const float py = static_cast<float>(y);
            const Vec3 rgb_a = (sample(px + dir_x * (-1.0f / 6.0f), py + dir_y * (-1.0f / 6.0f)) +
                sample(px + dir_x * (1.0f / 6.0f), py + dir_y * (1.0f / 6.0f))) * 0.5f;
            const Vec3 rgb_b = rgb_a * 0.5f +
                (sample(px + dir_x * -0.5f, py + dir_y * -0.5f) +
                 sample(px + dir_x * 0.5f, py + dir_y * 0.5f)) * 0.25f;
            const float luma_b = luminance(rgb_b);
            const Vec3 fxaa_color = (luma_b < lmin || luma_b > lmax) ? rgb_a : rgb_b;
            const float strength = geometry_edge ? 0.92f : std::clamp((range - 0.025f) / 0.20f, 0.0f, 0.75f);
            output[center_i] = lerp(input[center_i], fxaa_color, strength);
        }
    }
}

void apply_final_taa(const Scene& scene,
                     const RenderSettings& settings,
                     Framebuffer& framebuffer,
                     const std::vector<Vec3>& current_color) {
    const int width = framebuffer.width;
    const int height = framebuffer.height;
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    Framebuffer::TaaHistory& history = framebuffer.taa_history;
    if (history.color.size() != count) {
        history.resize(count);
    }
    if (settings.frame_index == 0u || invalidates_taa_history(settings.dirty)) {
        history.clear();
    }

    const bool use_taa = temporal_antialiasing_enabled(settings);
    const bool use_post_aa = post_antialiasing_enabled(settings);
    std::vector<Vec3> resolved(count);
    std::vector<float> resolved_history_length(count, 1.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            Vec3 color = current_color[index];
            if (use_taa) {
                Vec3 previous{};
                float previous_history_length = 0.0f;
                if (reproject_taa_history(framebuffer, settings, index, previous, previous_history_length)) {
                    previous = clip_history_ycocg(current_color, width, height, x, y, previous);
                    const float history_length = std::min(32.0f, previous_history_length + 1.0f);
                    const float min_alpha = settings.dirty == RenderDirty::None ? 0.04f : 0.25f;
                    const float alpha = std::max(min_alpha, 1.0f / history_length);
                    color = lerp(previous, current_color[index], alpha);
                    resolved_history_length[index] = history_length;
                }
            }
            resolved[index] = color;
        }
    }
    if (use_post_aa) {
        std::vector<Vec3> post_resolved;
        apply_luma_edge_post_aa(framebuffer.aov, width, height, resolved, post_resolved);
        resolved.swap(post_resolved);
    }
    for (size_t i = 0; i < count; ++i) {
        framebuffer.rgba[i] = to_rgba8(resolved[i]);
    }

    history.camera = scene.camera;
    history.jitter_x = settings.camera_jitter_x;
    history.jitter_y = settings.camera_jitter_y;
    history.valid = true;
    for (size_t i = 0; i < count; ++i) {
        history.color[i] = resolved[i];
        history.history_length[i] = resolved_history_length[i];
        history.normal[i] = framebuffer.aov.normal[i];
        history.world_position[i] = framebuffer.aov.world_position[i];
        history.linear_depth[i] = framebuffer.aov.linear_depth[i];
        history.object_id[i] = framebuffer.aov.object_id[i];
    }
}

} // namespace

void apply_svgf_denoiser(const Scene& scene, const RenderSettings& settings, Framebuffer& framebuffer) {
    (void)scene;
    const int width = framebuffer.width;
    const int height = framebuffer.height;
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (width <= 0 || height <= 0 || framebuffer.aov.radiance.size() != count || framebuffer.rgba.size() != count) {
        return;
    }

    Framebuffer::SvgfHistory& history = framebuffer.svgf_history;
    if (history.illumination.size() != count) {
        history.resize(count);
    }
    if (settings.frame_index == 0u || invalidates_svgf_history(settings.dirty)) {
        history.clear();
    }
    if (settings.frame_index == 0u || invalidates_taa_history(settings.dirty)) {
        framebuffer.taa_history.clear();
    }

    std::vector<Vec3> temporal_illumination(count);
    std::vector<float> temporal_moment1(count, 0.0f);
    std::vector<float> temporal_moment2(count, 0.0f);
    std::vector<float> temporal_variance(count, 0.0f);
    std::vector<float> current_history(count, 0.0f);

    for (size_t i = 0; i < count; ++i) {
        if (!has_valid_surface(framebuffer.aov, i)) {
            temporal_illumination[i] = framebuffer.aov.radiance[i];
            framebuffer.rgba[i] = to_rgba8(framebuffer.aov.radiance[i]);
            continue;
        }

        const Vec3 illumination = safe_demodulate(
            framebuffer.aov.radiance[i],
            framebuffer.aov.emission[i],
            framebuffer.aov.albedo[i]);
        const float lum = luminance(illumination);
        const float moment1 = lum;
        const float moment2 = lum * lum;

        Vec3 prev_illumination{};
        float prev_moment1 = 0.0f;
        float prev_moment2 = 0.0f;
        float prev_history = 0.0f;
        const bool reprojected = reproject_history(
            framebuffer, settings, i, prev_illumination, prev_moment1, prev_moment2, prev_history);
        const float history_length = std::min(32.0f, reprojected ? prev_history + 1.0f : 1.0f);
        const float alpha = reprojected ? std::max(settings.svgf_alpha, 1.0f / history_length) : 1.0f;
        const float moments_alpha = reprojected ? std::max(settings.svgf_moments_alpha, 1.0f / history_length) : 1.0f;

        temporal_illumination[i] = reprojected ? lerp(prev_illumination, illumination, alpha) : illumination;
        temporal_moment1[i] = reprojected ? lerp(prev_moment1, moment1, moments_alpha) : moment1;
        temporal_moment2[i] = reprojected ? lerp(prev_moment2, moment2, moments_alpha) : moment2;
        temporal_variance[i] = std::max(0.0f, temporal_moment2[i] - temporal_moment1[i] * temporal_moment1[i]);
        current_history[i] = history_length;
    }

    std::vector<Vec3> ping = temporal_illumination;
    std::vector<Vec3> pong(count);
    std::vector<float> variance_ping = temporal_variance;
    std::vector<float> variance_pong(count, 0.0f);
    const int iterations = std::clamp(settings.svgf_iterations, 0, 8);
    for (int i = 0; i < iterations; ++i) {
        atrous_pass(framebuffer.aov, ping, variance_ping, width, height, 1 << i, settings, pong, variance_pong);
        ping.swap(pong);
        variance_ping.swap(variance_pong);
    }

    std::vector<Vec3> final_color(count);
    history.camera = scene.camera;
    history.jitter_x = settings.camera_jitter_x;
    history.jitter_y = settings.camera_jitter_y;
    history.valid = true;
    for (size_t i = 0; i < count; ++i) {
        if (!has_valid_surface(framebuffer.aov, i)) {
            final_color[i] = framebuffer.aov.radiance[i];
            history.illumination[i] = framebuffer.aov.radiance[i];
            history.moment1[i] = luminance(framebuffer.aov.radiance[i]);
            history.moment2[i] = history.moment1[i] * history.moment1[i];
            history.history_length[i] = 0.0f;
            history.normal[i] = {};
            history.world_position[i] = {};
            history.linear_depth[i] = kInfinity;
            history.object_id[i] = 0u;
            continue;
        }

        const Vec3 color = debug_color(framebuffer.aov, ping, variance_ping, current_history, i, settings);
        final_color[i] = color;
        history.illumination[i] = ping[i];
        history.moment1[i] = temporal_moment1[i];
        history.moment2[i] = temporal_moment2[i];
        history.history_length[i] = current_history[i];
        history.normal[i] = framebuffer.aov.normal[i];
        history.world_position[i] = framebuffer.aov.world_position[i];
        history.linear_depth[i] = framebuffer.aov.linear_depth[i];
        history.object_id[i] = framebuffer.aov.object_id[i];
    }
    apply_final_taa(scene, settings, framebuffer, final_color);
}

} // namespace lt

#include "lt/material.h"

#include <algorithm>
#include <cmath>

namespace lt {
namespace {

Vec3 lerp(Vec3 a, Vec3 b, float t) {
    return a * (1.0f - t) + b * t;
}

Vec3 fresnel_schlick(float cos_theta, Vec3 f0) {
    const float f = std::pow(std::clamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
    return f0 + (Vec3{1.0f} - f0) * f;
}

Vec3 fresnel_conductor(float cos_theta, Vec3 eta, Vec3 k) {
    const float c = std::clamp(std::fabs(cos_theta), 0.0f, 1.0f);
    const float cos2 = c * c;
    const float sin2 = std::max(0.0f, 1.0f - cos2);
    const auto channel = [=](float e, float kk) {
        e = std::max(0.0f, e);
        kk = std::max(0.0f, kk);
        const float eta2 = e * e;
        const float k2 = kk * kk;
        const float t0 = eta2 - k2 - sin2;
        const float a2pb2 = std::sqrt(std::max(0.0f, t0 * t0 + 4.0f * eta2 * k2));
        const float a = std::sqrt(std::max(0.0f, 0.5f * (a2pb2 + t0)));
        const float t1 = a2pb2 + cos2;
        const float t2 = 2.0f * c * a;
        const float rs = (t1 - t2) / std::max(1.0e-6f, t1 + t2);
        const float t3 = cos2 * a2pb2 + sin2 * sin2;
        const float t4 = t2 * sin2;
        const float rp = rs * (t3 - t4) / std::max(1.0e-6f, t3 + t4);
        return std::clamp(0.5f * (rp + rs), 0.0f, 1.0f);
    };
    return {channel(eta.x, k.x), channel(eta.y, k.y), channel(eta.z, k.z)};
}

float ggx_distribution(float ndoth, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / std::max(1.0e-6f, kPi * d * d);
}

float ggx_distribution_alpha(float ndoth, float alpha) {
    const float a2 = alpha * alpha;
    const float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / std::max(1.0e-6f, kPi * d * d);
}

float smith_ggx_g1(float ndotv, float roughness) {
    const float r = roughness + 1.0f;
    const float k = (r * r) * 0.125f;
    return ndotv / std::max(1.0e-6f, ndotv * (1.0f - k) + k);
}

float smith_ggx_g1_alpha(float ndotv, float alpha) {
    if (ndotv <= 0.0f) {
        return 0.0f;
    }
    const float cos2 = ndotv * ndotv;
    const float tan2 = std::max(0.0f, 1.0f - cos2) / std::max(1.0e-6f, cos2);
    return 2.0f / (1.0f + std::sqrt(1.0f + alpha * alpha * tan2));
}

float charlie_sheen_distribution(float ndoth, float roughness) {
    const float alpha = std::clamp(roughness, 0.01f, 1.0f);
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - ndoth * ndoth));
    return (2.0f + 1.0f / alpha) * std::pow(sin_theta, 1.0f / alpha) / (2.0f * kPi);
}

float sheen_visibility(float ndotv, float ndotl) {
    return 1.0f / std::max(1.0e-6f, 4.0f * (ndotl + ndotv - ndotl * ndotv));
}

Vec3 reflect(Vec3 v, Vec3 n) {
    return n * (2.0f * dot(v, n)) - v;
}

Vec3 refract(Vec3 v, Vec3 n, float eta) {
    const float cos_theta = std::min(dot(v, n), 1.0f);
    const Vec3 perpendicular = (n * cos_theta - v) * eta;
    const float parallel_len2 = std::max(0.0f, 1.0f - dot(perpendicular, perpendicular));
    return normalize(perpendicular - n * std::sqrt(parallel_len2));
}

float fresnel_dielectric(float cos_theta, float eta) {
    float r0 = (1.0f - eta) / (1.0f + eta);
    r0 *= r0;
    return r0 + (1.0f - r0) * std::pow(std::clamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
}

Vec3 sample_ggx_half(Vec3 n, float roughness, float u1, float u2) {
    const float a = roughness * roughness;
    const float phi = 2.0f * kPi * u1;
    const float cos_theta = std::sqrt((1.0f - u2) / std::max(1.0e-6f, 1.0f + (a * a - 1.0f) * u2));
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
    return normalize(to_world({sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta}, n));
}

Vec3 sample_ggx_half_alpha(Vec3 n, float alpha, float u1, float u2) {
    const float a2 = alpha * alpha;
    const float phi = 2.0f * kPi * u1;
    const float cos_theta = std::sqrt((1.0f - u2) / std::max(1.0e-6f, 1.0f + (a2 - 1.0f) * u2));
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
    return normalize(to_world({sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta}, n));
}

float principled_specular_probability(float metallic) {
    return std::clamp(0.25f + 0.75f * metallic, 0.25f, 1.0f);
}

float max_channel(Vec3 v) {
    return std::max({v.x, v.y, v.z});
}

Vec2 transform_material_uv(const MaterialInput& input, Vec2 uv) {
    uv.x *= input.transform.scale.x;
    uv.y *= input.transform.scale.y;
    if (input.transform.rotation != 0.0f) {
        const float c = std::cos(input.transform.rotation);
        const float s = std::sin(input.transform.rotation);
        uv = {uv.x * c - uv.y * s, uv.x * s + uv.y * c};
    }
    uv.x += input.transform.offset.x;
    uv.y += input.transform.offset.y;
    return uv;
}

float material_input_scalar(const MaterialInput& input, Vec2 uv, float fallback = 1.0f) {
    if (!input.texture) {
        return input.scalar_factor * fallback;
    }
    uv = transform_material_uv(input, uv);
    if (input.channel == MaterialInputChannel::A) {
        return input.scalar_factor * input.texture->sample_alpha(uv);
    }
    const Vec3 sample = input.texture->sample(uv);
    float value = sample.x;
    if (input.channel == MaterialInputChannel::G) {
        value = sample.y;
    } else if (input.channel == MaterialInputChannel::B) {
        value = sample.z;
    } else if (input.channel == MaterialInputChannel::RGB) {
        value = (sample.x + sample.y + sample.z) / 3.0f;
    }
    return input.scalar_factor * value;
}

Vec3 material_input_color(const MaterialInput& input, Vec2 uv, Vec3 fallback = {1.0f, 1.0f, 1.0f}) {
    if (!input.texture) {
        return input.color_factor * fallback;
    }
    uv = transform_material_uv(input, uv);
    Vec3 sample = input.texture->sample(uv);
    if (input.channel == MaterialInputChannel::A) {
        sample = Vec3{input.texture->sample_alpha(uv)};
    } else if (input.channel == MaterialInputChannel::R) {
        sample = Vec3{sample.x};
    } else if (input.channel == MaterialInputChannel::G) {
        sample = Vec3{sample.y};
    } else if (input.channel == MaterialInputChannel::B) {
        sample = Vec3{sample.z};
    }
    return input.color_factor * sample;
}

Vec3 evaluate_standard_surface_lobes(
    Vec3 n,
    Vec3 wo,
    Vec3 wi,
    Vec3 color,
    float roughness,
    float metalness,
    float transmission,
    float specular_weight,
    float specular_ior,
    Vec3 sheen_color,
    float sheen_weight,
    float sheen_roughness,
    float coat,
    float coat_roughness) {
    const float ndotv = std::max(0.0f, dot(n, wo));
    const float ndotl = std::max(0.0f, dot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f) {
        return {};
    }

    const Vec3 h = normalize(wi + wo);
    const float ndoth = std::max(0.0f, dot(n, h));
    const float vdoth = std::max(0.0f, dot(wo, h));
    const float clamped_ior = std::clamp(specular_ior, 1.0f, 3.0f);
    const float dielectric_f0 = std::pow((1.0f - clamped_ior) / (1.0f + clamped_ior), 2.0f);
    const Vec3 f0 = lerp(Vec3{dielectric_f0 * specular_weight}, color, metalness);
    const Vec3 f = fresnel_schlick(vdoth, f0);
    const float d = ggx_distribution(ndoth, roughness);
    const float g = smith_ggx_g1(ndotv, roughness) * smith_ggx_g1(ndotl, roughness);
    const Vec3 specular = f * (d * g / std::max(1.0e-6f, 4.0f * ndotv * ndotl));
    const Vec3 diffuse = color * ((1.0f - metalness) * (1.0f - transmission)) / kPi;
    const Vec3 sheen = sheen_color * (sheen_weight * charlie_sheen_distribution(ndoth, sheen_roughness) * sheen_visibility(ndotv, ndotl));
    const Vec3 coat_f = fresnel_schlick(vdoth, Vec3{0.04f});
    const float coat_d = ggx_distribution(ndoth, coat_roughness);
    const float coat_g = smith_ggx_g1(ndotv, coat_roughness) * smith_ggx_g1(ndotl, coat_roughness);
    const Vec3 clearcoat_lobe = coat_f * (coat * coat_d * coat_g / std::max(1.0e-6f, 4.0f * ndotv * ndotl));
    return diffuse + specular + sheen + clearcoat_lobe;
}

} // namespace

Vec3 Material::base_color(Vec2 uv) const {
    return albedo_texture ? albedo * albedo_texture->sample(uv) : albedo;
}

float Material::opacity(Vec2 uv) const {
    const float texture_alpha = albedo_texture ? albedo_texture->sample_alpha(uv) : 1.0f;
    return std::clamp(alpha * texture_alpha, 0.0f, 1.0f);
}

Vec3 Material::emitted(Vec2 uv) const {
    return emission_texture ? emission * emission_texture->sample(uv) : emission;
}

float PrincipledMaterial::roughness_at(Vec2 uv) const {
    const float texture_roughness = metallic_roughness_texture ? metallic_roughness_texture->sample(uv).y : 1.0f;
    return std::clamp(roughness * texture_roughness, 0.02f, 1.0f);
}

float PrincipledMaterial::metallic_at(Vec2 uv) const {
    const float texture_metallic = metallic_roughness_texture ? metallic_roughness_texture->sample(uv).z : 1.0f;
    return std::clamp(metallic * texture_metallic, 0.0f, 1.0f);
}

Vec3 PrincipledMaterial::sheen_color_at(Vec2 uv) const {
    return sheen_color_texture ? sheen_color * sheen_color_texture->sample(uv) : sheen_color;
}

float PrincipledMaterial::sheen_roughness_at(Vec2 uv) const {
    const float texture_roughness = sheen_roughness_texture ? sheen_roughness_texture->sample_alpha(uv) : 1.0f;
    return std::clamp(sheen_roughness * texture_roughness, 0.01f, 1.0f);
}

float PrincipledMaterial::clearcoat_at(Vec2 uv) const {
    const float texture_clearcoat = clearcoat_texture ? clearcoat_texture->sample(uv).x : 1.0f;
    return std::clamp(clearcoat * texture_clearcoat, 0.0f, 1.0f);
}

float PrincipledMaterial::clearcoat_roughness_at(Vec2 uv) const {
    const float texture_roughness = clearcoat_roughness_texture ? clearcoat_roughness_texture->sample(uv).y : 1.0f;
    return std::clamp(clearcoat_roughness * texture_roughness, 0.02f, 1.0f);
}

Vec3 LambertianMaterial::evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = double_sided ? std::fabs(dot(n, wo)) : std::max(0.0f, dot(n, wo));
    const float ndotl = double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi));
    return ndotv > 0.0f && ndotl > 0.0f ? base_color(uv) / kPi : Vec3{};
}

float LambertianMaterial::pdf(Vec3 n, Vec3, Vec3 wi, Vec2) const {
    return (double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi))) / kPi;
}

MaterialSample LambertianMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng& rng) const {
    MaterialSample result;
    result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), n));
    result.pdf = pdf(n, wo, result.direction, uv);
    const float ndotl = std::max(0.0f, dot(n, result.direction));
    result.weight = result.pdf > 0.0f ? evaluate(n, wo, result.direction, uv) * (ndotl / result.pdf) : Vec3{};
    return result;
}

Vec3 DiffuseTransmissionMaterial::transmittance_color(Vec2 uv) const {
    return transmittance * (transmittance_texture ? transmittance_texture->sample(uv) : Vec3{1.0f});
}

float DiffuseTransmissionMaterial::transmission(Vec2 uv) const {
    return std::clamp(max_channel(transmittance_color(uv)), 0.0f, 1.0f);
}

Vec3 DiffuseTransmissionMaterial::evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = dot(n, wo);
    const float ndotl = dot(n, wi);
    if (ndotv == 0.0f || ndotl == 0.0f) {
        return {};
    }
    return ndotv * ndotl > 0.0f ? base_color(uv) / kPi : transmittance_color(uv) / kPi;
}

float DiffuseTransmissionMaterial::pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = dot(n, wo);
    const float ndotl = dot(n, wi);
    if (ndotv == 0.0f || ndotl == 0.0f) {
        return 0.0f;
    }
    const float reflect_weight = max_channel(base_color(uv));
    const float transmit_weight = max_channel(transmittance_color(uv));
    const float sum = reflect_weight + transmit_weight;
    const float transmit_prob = sum > 0.0f ? transmit_weight / sum : 0.5f;
    const bool transmission_lobe = ndotv * ndotl < 0.0f;
    const float lobe_prob = transmission_lobe ? transmit_prob : 1.0f - transmit_prob;
    return lobe_prob * std::fabs(ndotl) / kPi;
}

MaterialSample DiffuseTransmissionMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng& rng) const {
    MaterialSample result;
    const Vec3 reflectance = base_color(uv);
    const float reflect_weight = max_channel(reflectance);
    const float transmit_weight = max_channel(transmittance_color(uv));
    const float sum = reflect_weight + transmit_weight;
    const float transmit_prob = sum > 0.0f ? transmit_weight / sum : 0.5f;
    const Vec3 facing_n = dot(n, wo) >= 0.0f ? n : -n;
    const bool sample_transmission = rng.next_float() < transmit_prob;
    const Vec3 sample_n = sample_transmission ? -facing_n : facing_n;
    result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), sample_n));
    result.pdf = pdf(n, wo, result.direction, uv);
    const float ndotl = std::fabs(dot(n, result.direction));
    result.weight = result.pdf > 0.0f ? evaluate(n, wo, result.direction, uv) * (ndotl / result.pdf) : Vec3{};
    return result;
}

Vec3 PrincipledMaterial::evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = double_sided ? std::fabs(dot(n, wo)) : std::max(0.0f, dot(n, wo));
    const float ndotl = double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f) {
        return {};
    }

    const float r = roughness_at(uv);
    const float m = metallic_at(uv);
    const float sheen_r = sheen_roughness_at(uv);
    const float coat = clearcoat_at(uv);
    const float coat_r = clearcoat_roughness_at(uv);
    const Vec3 color = base_color(uv);
    const Vec3 h = normalize(wi + wo);
    const float ndoth = std::max(0.0f, dot(n, h));
    const float vdoth = std::max(0.0f, dot(wo, h));
    const Vec3 f0 = lerp(Vec3{0.04f}, color, m);
    const Vec3 f = fresnel_schlick(vdoth, f0);
    const float d = ggx_distribution(ndoth, r);
    const float g = smith_ggx_g1(ndotv, r) * smith_ggx_g1(ndotl, r);
    const Vec3 specular = f * (d * g / std::max(1.0e-6f, 4.0f * ndotv * ndotl));
    const Vec3 diffuse = color * (1.0f - m) / kPi;
    const Vec3 sheen = sheen_color_at(uv) * (charlie_sheen_distribution(ndoth, sheen_r) * sheen_visibility(ndotv, ndotl));
    const Vec3 coat_f = fresnel_schlick(vdoth, Vec3{0.04f});
    const float coat_d = ggx_distribution(ndoth, coat_r);
    const float coat_g = smith_ggx_g1(ndotv, coat_r) * smith_ggx_g1(ndotl, coat_r);
    const Vec3 clearcoat_lobe = coat_f * (coat * coat_d * coat_g / std::max(1.0e-6f, 4.0f * ndotv * ndotl));
    return diffuse + specular + sheen + clearcoat_lobe;
}

float PrincipledMaterial::pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotl = double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi));
    const float ndotv = double_sided ? std::fabs(dot(n, wo)) : std::max(0.0f, dot(n, wo));
    if (ndotl <= 0.0f || ndotv <= 0.0f) {
        return 0.0f;
    }
    const float r = roughness_at(uv);
    const float spec_prob = std::clamp(principled_specular_probability(metallic_at(uv)) + 0.25f * clearcoat_at(uv), 0.25f, 1.0f);
    const Vec3 h = normalize(wi + wo);
    const float ndoth = std::max(0.0f, dot(n, h));
    const float vdoth = std::max(0.0f, dot(wo, h));
    const float diffuse_pdf = ndotl / kPi;
    const float specular_pdf = ggx_distribution(ndoth, r) * ndoth / std::max(1.0e-6f, 4.0f * vdoth);
    return (1.0f - spec_prob) * diffuse_pdf + spec_prob * specular_pdf;
}

MaterialSample PrincipledMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng& rng) const {
    MaterialSample result;
    const float r = roughness_at(uv);
    const float m = metallic_at(uv);
    const float spec_prob = std::clamp(principled_specular_probability(m) + 0.25f * clearcoat_at(uv), 0.25f, 1.0f);
    if (rng.next_float() < spec_prob) {
        const Vec3 h = sample_ggx_half(n, r, rng.next_float(), rng.next_float());
        result.direction = normalize(reflect(wo, h));
        if (dot(result.direction, n) <= 0.0f) {
            result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), n));
        }
    } else {
        result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), n));
    }
    result.pdf = pdf(n, wo, result.direction, uv);
    const float ndotl = std::max(0.0f, dot(n, result.direction));
    result.weight = result.pdf > 0.0f ? evaluate(n, wo, result.direction, uv) * (ndotl / result.pdf) : Vec3{};
    return result;
}

MaterialSample MirrorMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng&) const {
    MaterialSample result;
    result.direction = normalize(reflect(wo, n));
    (void)uv;
    result.weight = {1.0f, 1.0f, 1.0f};
    result.pdf = 1.0f;
    result.delta = true;
    return result;
}

MaterialSample DielectricMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const {
    MaterialSample result;
    const float eta = front_face ? 1.0f / std::max(1.0f, ior) : std::max(1.0f, ior);
    const float cos_theta = std::min(dot(wo, n), 1.0f);
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
    const bool cannot_refract = eta * sin_theta > 1.0f;
    const float reflectance = fresnel_dielectric(cos_theta, eta);
    result.direction = (cannot_refract || reflectance > rng.next_float()) ? normalize(reflect(wo, n)) : refract(wo, n, eta);
    result.weight = transmission_tint;
    (void)uv;
    result.pdf = 1.0f;
    result.delta = true;
    return result;
}

float ConductorMaterial::roughness_at(Vec2) const {
    return std::clamp(roughness, 0.0f, 1.0f);
}

Vec3 ConductorMaterial::evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = double_sided ? std::fabs(dot(n, wo)) : std::max(0.0f, dot(n, wo));
    const float ndotl = double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f || roughness_at(uv) <= 0.001f) {
        return {};
    }
    const Vec3 facing_n = dot(n, wo) >= 0.0f ? n : -n;
    const Vec3 h = normalize(wi + wo);
    const float ndoth = std::max(0.0f, dot(facing_n, h));
    const float vdoth = std::max(0.0f, dot(wo, h));
    if (ndoth <= 0.0f || vdoth <= 0.0f) {
        return {};
    }
    const float micro_alpha = std::clamp(roughness_at(uv), 0.001f, 1.0f);
    const Vec3 f = base_color(uv) * fresnel_conductor(vdoth, eta, k);
    const float d = ggx_distribution_alpha(ndoth, micro_alpha);
    const float g = smith_ggx_g1_alpha(ndotv, micro_alpha) * smith_ggx_g1_alpha(ndotl, micro_alpha);
    return f * (d * g / std::max(1.0e-6f, 4.0f * ndotv * ndotl));
}

float ConductorMaterial::pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = double_sided ? std::fabs(dot(n, wo)) : std::max(0.0f, dot(n, wo));
    const float ndotl = double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f || roughness_at(uv) <= 0.001f) {
        return 0.0f;
    }
    const Vec3 facing_n = dot(n, wo) >= 0.0f ? n : -n;
    const Vec3 h = normalize(wi + wo);
    const float ndoth = std::max(0.0f, dot(facing_n, h));
    const float vdoth = std::max(0.0f, dot(wo, h));
    if (ndoth <= 0.0f || vdoth <= 0.0f) {
        return 0.0f;
    }
    const float micro_alpha = std::clamp(roughness_at(uv), 0.001f, 1.0f);
    return ggx_distribution_alpha(ndoth, micro_alpha) * ndoth / std::max(1.0e-6f, 4.0f * vdoth);
}

MaterialSample ConductorMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng& rng) const {
    MaterialSample result;
    const Vec3 facing_n = dot(n, wo) >= 0.0f ? n : -n;
    const float micro_alpha = roughness_at(uv);
    if (micro_alpha <= 0.001f) {
        result.direction = normalize(reflect(wo, facing_n));
        result.weight = base_color(uv) * fresnel_conductor(dot(facing_n, wo), eta, k);
        result.pdf = 1.0f;
        result.delta = true;
        return result;
    }

    const Vec3 h = sample_ggx_half_alpha(facing_n, std::clamp(micro_alpha, 0.001f, 1.0f), rng.next_float(), rng.next_float());
    result.direction = normalize(reflect(wo, h));
    if (dot(result.direction, facing_n) <= 0.0f) {
        result.pdf = 0.0f;
        result.weight = {};
        return result;
    }
    result.pdf = pdf(n, wo, result.direction, uv);
    const float ndotl = std::max(0.0f, dot(facing_n, result.direction));
    result.weight = result.pdf > 0.0f ? evaluate(n, wo, result.direction, uv) * (ndotl / result.pdf) : Vec3{};
    return result;
}

Vec3 StandardSurfaceMaterial::standard_base_color(Vec2 uv) const {
    if (base_color_input.texture) {
        return albedo * material_input_color(base_color_input, uv);
    }
    return base_color(uv);
}

Vec3 StandardSurfaceMaterial::emitted(Vec2 uv) const {
    if (emission_input.texture) {
        return emission * material_input_color(emission_input, uv);
    }
    return Material::emitted(uv);
}

float StandardSurfaceMaterial::opacity(Vec2 uv) const {
    return std::clamp(Material::opacity(uv) * material_input_scalar(opacity_input, uv), 0.0f, 1.0f);
}

float StandardSurfaceMaterial::roughness_at(Vec2 uv) const {
    return std::clamp(roughness * material_input_scalar(roughness_input, uv), 0.02f, 1.0f);
}

float StandardSurfaceMaterial::metalness_at(Vec2 uv) const {
    return std::clamp(metalness * material_input_scalar(metalness_input, uv), 0.0f, 1.0f);
}

float StandardSurfaceMaterial::specular_weight_at(Vec2 uv) const {
    return std::max(0.0f, specular_weight * material_input_scalar(specular_weight_input, uv));
}

float StandardSurfaceMaterial::transmission(Vec2 uv) const {
    return std::clamp(transmission_weight * material_input_scalar(transmission_input, uv), 0.0f, 1.0f);
}

float StandardSurfaceMaterial::coat_at(Vec2 uv) const {
    return std::clamp(coat_weight * material_input_scalar(coat_input, uv), 0.0f, 1.0f);
}

float StandardSurfaceMaterial::coat_roughness_at(Vec2 uv) const {
    return std::clamp(coat_roughness * material_input_scalar(coat_roughness_input, uv), 0.02f, 1.0f);
}

Vec3 StandardSurfaceMaterial::sheen_color_at(Vec2 uv) const {
    return sheen_color_input.texture ? material_input_color(sheen_color_input, uv, sheen_color) : sheen_color;
}

float StandardSurfaceMaterial::sheen_roughness_at(Vec2 uv) const {
    return std::clamp(sheen_roughness * material_input_scalar(sheen_roughness_input, uv), 0.01f, 1.0f);
}

Vec3 StandardSurfaceMaterial::evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = double_sided ? std::fabs(dot(n, wo)) : std::max(0.0f, dot(n, wo));
    const float ndotl = double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f) {
        return {};
    }
    const Vec3 facing_n = dot(n, wo) >= 0.0f ? n : -n;
    return evaluate_standard_surface_lobes(
        facing_n,
        wo,
        wi,
        standard_base_color(uv),
        roughness_at(uv),
        metalness_at(uv),
        transmission(uv),
        specular_weight_at(uv),
        specular_ior,
        sheen_color_at(uv),
        sheen_weight,
        sheen_roughness_at(uv),
        coat_at(uv),
        coat_roughness_at(uv));
}

float StandardSurfaceMaterial::pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotl = double_sided ? std::fabs(dot(n, wi)) : std::max(0.0f, dot(n, wi));
    const float ndotv = double_sided ? std::fabs(dot(n, wo)) : std::max(0.0f, dot(n, wo));
    if (ndotl <= 0.0f || ndotv <= 0.0f) {
        return 0.0f;
    }
    const float transmission_prob = transmission(uv);
    if (transmission_prob >= 0.999f) {
        return 1.0f;
    }
    const Vec3 facing_n = dot(n, wo) >= 0.0f ? n : -n;
    const float diffuse_pdf = std::max(0.0f, dot(facing_n, wi)) / kPi;
    const float r = roughness_at(uv);
    const float m = metalness_at(uv);
    const float spec_prob = std::clamp(0.05f + 0.20f * specular_weight_at(uv) + 0.75f * m + 0.25f * coat_at(uv), 0.05f, 1.0f);
    const Vec3 h = normalize(wi + wo);
    const float ndoth = std::max(0.0f, dot(facing_n, h));
    const float vdoth = std::max(0.0f, dot(wo, h));
    const float specular_pdf = ggx_distribution(ndoth, r) * ndoth / std::max(1.0e-6f, 4.0f * vdoth);
    return transmission_prob + (1.0f - transmission_prob) * ((1.0f - spec_prob) * diffuse_pdf + spec_prob * specular_pdf);
}

MaterialSample StandardSurfaceMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const {
    MaterialSample result;
    const float trans = transmission(uv);
    const Vec3 color = standard_base_color(uv);
    (void)color;
    const Vec3 tint = transmission_color;
    if (trans > 0.0f && rng.next_float() < trans) {
        const float ior = std::clamp(specular_ior, 1.0f, 3.0f);
        const float eta = front_face ? 1.0f / ior : ior;
        const float cos_theta = std::min(dot(wo, n), 1.0f);
        const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
        const bool cannot_refract = eta * sin_theta > 1.0f;
        const float reflectance = fresnel_dielectric(cos_theta, eta);
        result.direction = (cannot_refract || reflectance > rng.next_float()) ? normalize(reflect(wo, n)) : refract(wo, n, eta);
        result.weight = tint;
        result.pdf = std::max(1.0e-6f, trans);
        result.delta = true;
        return result;
    }

    const float r = roughness_at(uv);
    const float m = metalness_at(uv);
    const float spec_prob = std::clamp(0.05f + 0.20f * specular_weight_at(uv) + 0.75f * m + 0.25f * coat_at(uv), 0.05f, 1.0f);
    const Vec3 facing_n = dot(n, wo) >= 0.0f ? n : -n;
    if (rng.next_float() < spec_prob) {
        const Vec3 h = sample_ggx_half(facing_n, r, rng.next_float(), rng.next_float());
        result.direction = normalize(reflect(wo, h));
        if (dot(result.direction, facing_n) <= 0.0f) {
            result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), facing_n));
        }
    } else {
        result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), facing_n));
    }
    result.pdf = pdf(n, wo, result.direction, uv);
    const float ndotl = std::max(0.0f, dot(facing_n, result.direction));
    result.weight = result.pdf > 0.0f ? evaluate(n, wo, result.direction, uv) * (ndotl / result.pdf) : Vec3{};
    return result;
}

std::shared_ptr<Material> make_material(const std::string& name, Vec3 albedo, BrdfModel model, float roughness, float metallic) {
    if (model == BrdfModel::StandardSurface) {
        return std::make_shared<StandardSurfaceMaterial>(name, albedo, std::clamp(roughness, 0.02f, 1.0f), std::clamp(metallic, 0.0f, 1.0f));
    }
    if (model == BrdfModel::DiffuseTransmission) {
        return std::make_shared<DiffuseTransmissionMaterial>(name, albedo, Vec3{std::clamp(metallic, 0.0f, 1.0f)});
    }
    if (model == BrdfModel::Principled) {
        return std::make_shared<PrincipledMaterial>(name, albedo, std::clamp(roughness, 0.02f, 1.0f), std::clamp(metallic, 0.0f, 1.0f));
    }
    if (model == BrdfModel::Mirror) {
        return std::make_shared<MirrorMaterial>(name, albedo);
    }
    if (model == BrdfModel::Dielectric) {
        return std::make_shared<DielectricMaterial>(name, albedo, std::clamp(roughness, 1.0f, 3.0f));
    }
    if (model == BrdfModel::Conductor) {
        return std::make_shared<ConductorMaterial>(name, albedo, std::clamp(roughness, 0.0f, 1.0f));
    }
    return std::make_shared<LambertianMaterial>(name, albedo);
}

BrdfModel parse_brdf_model(const std::string& name) {
    if (name == "standard_surface" || name == "standard-surface" || name == "openpbr" || name == "open_pbr") return BrdfModel::StandardSurface;
    if (name == "diffuse_transmission" || name == "diffuse-transmission" || name == "diffusetransmission") return BrdfModel::DiffuseTransmission;
    if (name == "principled" || name == "cook_torrance" || name == "cook-torrance" || name == "ggx") return BrdfModel::Principled;
    if (name == "mirror" || name == "specular") return BrdfModel::Mirror;
    if (name == "dielectric" || name == "glass") return BrdfModel::Dielectric;
    if (name == "conductor" || name == "metal") return BrdfModel::Conductor;
    return BrdfModel::Lambertian;
}

NprStyle parse_npr_style(const std::string& name) {
    if (name == "color_map" || name == "color-map" || name == "colormap" || name == "color") {
        return NprStyle::ColorMap;
    }
    if (name == "x_toon" || name == "x-toon" || name == "xtoon" || name == "toon") {
        return NprStyle::XToon;
    }
    if (name == "cross_hatching" || name == "cross-hatching" || name == "crosshatching" ||
        name == "cross_hatch" || name == "cross-hatch" || name == "crosshatch" ||
        name == "hatching" || name == "hatch") {
        return NprStyle::CrossHatching;
    }
    return NprStyle::None;
}

const char* npr_style_name(NprStyle style) {
    switch (style) {
    case NprStyle::ColorMap: return "color_map";
    case NprStyle::XToon: return "x_toon";
    case NprStyle::CrossHatching: return "cross_hatching";
    case NprStyle::None:
    default:
        return "none";
    }
}

XToonDetailMode parse_xtoon_detail_mode(const std::string& name) {
    if (name == "depth" || name == "distance") {
        return XToonDetailMode::Depth;
    }
    if (name == "silhouette" || name == "near_silhouette" || name == "near-silhouette" || name == "rim") {
        return XToonDetailMode::NearSilhouette;
    }
    if (name == "highlight" || name == "specular") {
        return XToonDetailMode::Highlight;
    }
    return XToonDetailMode::Constant;
}

const char* xtoon_detail_mode_name(XToonDetailMode mode) {
    switch (mode) {
    case XToonDetailMode::Depth: return "depth";
    case XToonDetailMode::NearSilhouette: return "silhouette";
    case XToonDetailMode::Highlight: return "highlight";
    case XToonDetailMode::Constant:
    default:
        return "constant";
    }
}

} // namespace lt

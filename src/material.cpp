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

float ggx_distribution(float ndoth, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / std::max(1.0e-6f, kPi * d * d);
}

float smith_ggx_g1(float ndotv, float roughness) {
    const float r = roughness + 1.0f;
    const float k = (r * r) * 0.125f;
    return ndotv / std::max(1.0e-6f, ndotv * (1.0f - k) + k);
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

float principled_specular_probability(float metallic) {
    return std::clamp(0.25f + 0.75f * metallic, 0.25f, 1.0f);
}

} // namespace

Vec3 Material::base_color(Vec2 uv) const {
    return albedo_texture ? albedo * albedo_texture->sample(uv) : albedo;
}

Vec3 LambertianMaterial::evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    return dot(n, wo) > 0.0f && dot(n, wi) > 0.0f ? base_color(uv) / kPi : Vec3{};
}

float LambertianMaterial::pdf(Vec3 n, Vec3, Vec3 wi) const {
    return std::max(0.0f, dot(n, wi)) / kPi;
}

MaterialSample LambertianMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng& rng) const {
    MaterialSample result;
    result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), n));
    result.pdf = pdf(n, wo, result.direction);
    const float ndotl = std::max(0.0f, dot(n, result.direction));
    result.weight = result.pdf > 0.0f ? evaluate(n, wo, result.direction, uv) * (ndotl / result.pdf) : Vec3{};
    return result;
}

Vec3 PrincipledMaterial::evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const {
    const float ndotv = std::max(0.0f, dot(n, wo));
    const float ndotl = std::max(0.0f, dot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f) {
        return {};
    }

    const float r = std::clamp(roughness, 0.02f, 1.0f);
    const float m = std::clamp(metallic, 0.0f, 1.0f);
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
    return diffuse + specular;
}

float PrincipledMaterial::pdf(Vec3 n, Vec3 wo, Vec3 wi) const {
    const float ndotl = std::max(0.0f, dot(n, wi));
    const float ndotv = std::max(0.0f, dot(n, wo));
    if (ndotl <= 0.0f || ndotv <= 0.0f) {
        return 0.0f;
    }
    const float r = std::clamp(roughness, 0.02f, 1.0f);
    const float spec_prob = principled_specular_probability(std::clamp(metallic, 0.0f, 1.0f));
    const Vec3 h = normalize(wi + wo);
    const float ndoth = std::max(0.0f, dot(n, h));
    const float vdoth = std::max(0.0f, dot(wo, h));
    const float diffuse_pdf = ndotl / kPi;
    const float specular_pdf = ggx_distribution(ndoth, r) * ndoth / std::max(1.0e-6f, 4.0f * vdoth);
    return (1.0f - spec_prob) * diffuse_pdf + spec_prob * specular_pdf;
}

MaterialSample PrincipledMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng& rng) const {
    MaterialSample result;
    const float m = std::clamp(metallic, 0.0f, 1.0f);
    const float spec_prob = principled_specular_probability(m);
    if (rng.next_float() < spec_prob) {
        const Vec3 h = sample_ggx_half(n, std::clamp(roughness, 0.02f, 1.0f), rng.next_float(), rng.next_float());
        result.direction = normalize(reflect(wo, h));
        if (dot(result.direction, n) <= 0.0f) {
            result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), n));
        }
    } else {
        result.direction = normalize(to_world(cosine_sample_hemisphere(rng.next_float(), rng.next_float()), n));
    }
    result.pdf = pdf(n, wo, result.direction);
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
    result.weight = base_color(uv);
    result.pdf = 1.0f;
    result.delta = true;
    return result;
}

MaterialSample ConductorMaterial::sample(Vec3 n, Vec3 wo, Vec2 uv, bool, Rng&) const {
    MaterialSample result;
    result.direction = normalize(reflect(wo, n));
    result.weight = base_color(uv);
    result.pdf = 1.0f;
    result.delta = true;
    return result;
}

std::shared_ptr<Material> make_material(const std::string& name, Vec3 albedo, BrdfModel model, float roughness, float metallic) {
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
        return std::make_shared<ConductorMaterial>(name, albedo);
    }
    return std::make_shared<LambertianMaterial>(name, albedo);
}

BrdfModel parse_brdf_model(const std::string& name) {
    if (name == "principled" || name == "cook_torrance" || name == "cook-torrance" || name == "ggx") return BrdfModel::Principled;
    if (name == "mirror" || name == "specular") return BrdfModel::Mirror;
    if (name == "dielectric" || name == "glass") return BrdfModel::Dielectric;
    if (name == "conductor" || name == "metal") return BrdfModel::Conductor;
    return BrdfModel::Lambertian;
}

} // namespace lt

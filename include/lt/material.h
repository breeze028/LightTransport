#pragma once

#include "lt/math.h"
#include "lt/texture.h"

#include <memory>
#include <string>
#include <utility>

namespace lt {

enum class BrdfModel {
    Lambertian = 0,
    Principled = 1,
    Mirror = 2,
    Dielectric = 3,
    Conductor = 4,
};

enum class AlphaMode {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
};

enum class NprStyle {
    None = 0,
    ColorMap = 1,
    XToon = 2,
    CrossHatching = 3,
};

enum class XToonDetailMode {
    Constant = 0,
    Depth = 1,
    NearSilhouette = 2,
    Highlight = 3,
};

struct NprSettings {
    NprStyle style = NprStyle::None;
    float value_min = 0.0f;
    float value_max = 1.0f;
    XToonDetailMode xtoon_detail_mode = XToonDetailMode::Highlight;
    int xtoon_steps = 3;
    Vec3 xtoon_shadow = {0.45f, 0.45f, 0.55f};
    Vec3 xtoon_mid = {0.82f, 0.82f, 0.90f};
    Vec3 xtoon_lit = {1.16f, 1.14f, 1.05f};
    Vec3 xtoon_accent = {1.0f, 0.92f, 0.68f};
    float xtoon_detail_strength = 0.75f;
    float xtoon_detail_threshold = 0.18f;
    float xtoon_detail_power = 24.0f;
    float xtoon_depth_near = 1.0f;
    float xtoon_depth_far = 8.0f;
    int hatch_sets = 3;
    float hatch_spacing = 0.08f;
    float hatch_width = 0.012f;
    float hatch_angle = 0.0f;
    float hatch_value_min = 0.0f;
    float hatch_value_max = 1.0f;
    Vec3 hatch_ink = {0.05f, 0.045f, 0.04f};
    Vec3 hatch_paper = {0.96f, 0.94f, 0.88f};
    bool hatch_passthrough = false;
    bool hatch_shadow_only = false;
};

struct MaterialSample {
    Vec3 direction;
    Vec3 weight;
    float pdf = 0.0f;
    bool delta = false;
};

class Material {
public:
    std::string name = "mat";
    Vec3 albedo = {0.8f, 0.8f, 0.8f};
    float alpha = 1.0f;
    float alpha_cutoff = 0.5f;
    AlphaMode alpha_mode = AlphaMode::Opaque;
    bool double_sided = false;
    std::shared_ptr<Texture> albedo_texture;
    std::shared_ptr<Texture> normal_texture;
    float normal_scale = 1.0f;
    Vec3 emission = {};
    std::shared_ptr<Texture> emission_texture;
    NprSettings npr;

    Material() = default;
    Material(std::string name_, Vec3 albedo_) : name(std::move(name_)), albedo(albedo_) {}
    virtual ~Material() = default;

    virtual BrdfModel model() const = 0;
    virtual const char* model_name() const = 0;
    Vec3 base_color(Vec2 uv) const;
    float opacity(Vec2 uv) const;
    Vec3 emitted(Vec2 uv) const;
    virtual Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const = 0;
    virtual float pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const = 0;
    virtual MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const = 0;
    virtual std::shared_ptr<Material> clone() const = 0;
};

class LambertianMaterial final : public Material {
public:
    using Material::Material;

    BrdfModel model() const override { return BrdfModel::Lambertian; }
    const char* model_name() const override { return "lambertian"; }
    Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    float pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<LambertianMaterial>(*this); }
};

class PrincipledMaterial final : public Material {
public:
    float roughness = 0.5f;
    float metallic = 0.0f;
    std::shared_ptr<Texture> metallic_roughness_texture;
    Vec3 sheen_color = {};
    float sheen_roughness = 0.0f;
    std::shared_ptr<Texture> sheen_color_texture;
    std::shared_ptr<Texture> sheen_roughness_texture;
    float clearcoat = 0.0f;
    float clearcoat_roughness = 0.0f;
    std::shared_ptr<Texture> clearcoat_texture;
    std::shared_ptr<Texture> clearcoat_roughness_texture;

    PrincipledMaterial() = default;
    PrincipledMaterial(std::string name_, Vec3 albedo_, float roughness_, float metallic_)
        : Material(std::move(name_), albedo_), roughness(roughness_), metallic(metallic_) {}

    BrdfModel model() const override { return BrdfModel::Principled; }
    const char* model_name() const override { return "principled"; }
    float roughness_at(Vec2 uv) const;
    float metallic_at(Vec2 uv) const;
    Vec3 sheen_color_at(Vec2 uv) const;
    float sheen_roughness_at(Vec2 uv) const;
    float clearcoat_at(Vec2 uv) const;
    float clearcoat_roughness_at(Vec2 uv) const;
    Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    float pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<PrincipledMaterial>(*this); }
};

class MirrorMaterial final : public Material {
public:
    using Material::Material;

    BrdfModel model() const override { return BrdfModel::Mirror; }
    const char* model_name() const override { return "mirror"; }
    Vec3 evaluate(Vec3, Vec3, Vec3, Vec2) const override { return {}; }
    float pdf(Vec3, Vec3, Vec3, Vec2) const override { return 0.0f; }
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<MirrorMaterial>(*this); }
};

class DielectricMaterial final : public Material {
public:
    float ior = 1.5f;
    Vec3 transmission_tint = {1.0f, 1.0f, 1.0f};

    DielectricMaterial() = default;
    DielectricMaterial(std::string name_, Vec3 albedo_, float ior_)
        : Material(std::move(name_), albedo_), ior(ior_) {}

    BrdfModel model() const override { return BrdfModel::Dielectric; }
    const char* model_name() const override { return "dielectric"; }
    Vec3 evaluate(Vec3, Vec3, Vec3, Vec2) const override { return {}; }
    float pdf(Vec3, Vec3, Vec3, Vec2) const override { return 0.0f; }
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<DielectricMaterial>(*this); }
};

class ConductorMaterial final : public Material {
public:
    using Material::Material;

    BrdfModel model() const override { return BrdfModel::Conductor; }
    const char* model_name() const override { return "conductor"; }
    Vec3 evaluate(Vec3, Vec3, Vec3, Vec2) const override { return {}; }
    float pdf(Vec3, Vec3, Vec3, Vec2) const override { return 0.0f; }
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<ConductorMaterial>(*this); }
};

std::shared_ptr<Material> make_material(const std::string& name, Vec3 albedo, BrdfModel model, float roughness, float metallic);
BrdfModel parse_brdf_model(const std::string& name);
NprStyle parse_npr_style(const std::string& name);
const char* npr_style_name(NprStyle style);
XToonDetailMode parse_xtoon_detail_mode(const std::string& name);
const char* xtoon_detail_mode_name(XToonDetailMode mode);

} // namespace lt

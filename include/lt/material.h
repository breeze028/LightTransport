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
    StandardSurface = 5,
    DiffuseTransmission = 6,
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

enum class MaterialInputChannel {
    RGB = 0,
    R = 1,
    G = 2,
    B = 3,
    A = 4,
};

struct TextureTransform {
    Vec2 offset = {};
    Vec2 scale = {1.0f, 1.0f};
    float rotation = 0.0f;
    int uv_set = 0;
};

struct MaterialInput {
    std::shared_ptr<Texture> texture;
    Vec3 color_factor = {1.0f, 1.0f, 1.0f};
    float scalar_factor = 1.0f;
    MaterialInputChannel channel = MaterialInputChannel::RGB;
    TextureColorSpace color_space = TextureColorSpace::Auto;
    TextureRole role = TextureRole::Unknown;
    TextureTransform transform;
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
    virtual float opacity(Vec2 uv) const;
    virtual Vec3 emitted(Vec2 uv) const;
    virtual float transmission(Vec2) const { return 0.0f; }
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

class DiffuseTransmissionMaterial final : public Material {
public:
    Vec3 transmittance = {0.5f, 0.5f, 0.5f};
    std::shared_ptr<Texture> transmittance_texture;

    DiffuseTransmissionMaterial() = default;
    DiffuseTransmissionMaterial(std::string name_, Vec3 reflectance_, Vec3 transmittance_)
        : Material(std::move(name_), reflectance_), transmittance(transmittance_) {}

    BrdfModel model() const override { return BrdfModel::DiffuseTransmission; }
    const char* model_name() const override { return "diffuse_transmission"; }
    Vec3 transmittance_color(Vec2 uv) const;
    float transmission(Vec2 uv) const override;
    Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    float pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<DiffuseTransmissionMaterial>(*this); }
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
    float roughness = 0.0f;
    Vec3 eta = {0.200438f, 0.924033f, 1.102212f};
    Vec3 k = {3.912949f, 2.452848f, 2.142188f};

    ConductorMaterial() = default;
    ConductorMaterial(std::string name_, Vec3 albedo_, float roughness_ = 0.0f, Vec3 eta_ = {0.200438f, 0.924033f, 1.102212f}, Vec3 k_ = {3.912949f, 2.452848f, 2.142188f})
        : Material(std::move(name_), albedo_), roughness(roughness_), eta(eta_), k(k_) {}

    BrdfModel model() const override { return BrdfModel::Conductor; }
    const char* model_name() const override { return "conductor"; }
    float roughness_at(Vec2 uv) const;
    Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    float pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<ConductorMaterial>(*this); }
};

class StandardSurfaceMaterial final : public Material {
public:
    float roughness = 0.5f;
    float metalness = 0.0f;
    float specular_weight = 1.0f;
    float specular_ior = 1.5f;
    float transmission_weight = 0.0f;
    Vec3 transmission_color = {1.0f, 1.0f, 1.0f};
    float coat_weight = 0.0f;
    float coat_roughness = 0.0f;
    Vec3 sheen_color = {};
    float sheen_weight = 0.0f;
    float sheen_roughness = 0.0f;
    float subsurface_weight = 0.0f;
    Vec3 subsurface_color = {1.0f, 1.0f, 1.0f};
    float volume_density = 0.0f;
    Vec3 volume_color = {1.0f, 1.0f, 1.0f};
    bool thin_walled = false;
    bool unsupported_volume = false;
    bool unsupported_subsurface = false;

    MaterialInput base_color_input;
    MaterialInput roughness_input;
    MaterialInput metalness_input;
    MaterialInput specular_weight_input;
    MaterialInput transmission_input;
    MaterialInput opacity_input;
    MaterialInput emission_input;
    MaterialInput coat_input;
    MaterialInput coat_roughness_input;
    MaterialInput sheen_color_input;
    MaterialInput sheen_roughness_input;

    StandardSurfaceMaterial() = default;
    StandardSurfaceMaterial(std::string name_, Vec3 albedo_, float roughness_, float metalness_)
        : Material(std::move(name_), albedo_), roughness(roughness_), metalness(metalness_) {}

    BrdfModel model() const override { return BrdfModel::StandardSurface; }
    const char* model_name() const override { return "standard_surface"; }
    Vec3 standard_base_color(Vec2 uv) const;
    Vec3 emitted(Vec2 uv) const override;
    float opacity(Vec2 uv) const override;
    float roughness_at(Vec2 uv) const;
    float metalness_at(Vec2 uv) const;
    float specular_weight_at(Vec2 uv) const;
    float transmission(Vec2 uv) const override;
    float coat_at(Vec2 uv) const;
    float coat_roughness_at(Vec2 uv) const;
    Vec3 sheen_color_at(Vec2 uv) const;
    float sheen_roughness_at(Vec2 uv) const;
    Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    float pdf(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<StandardSurfaceMaterial>(*this); }
};

std::shared_ptr<Material> make_material(const std::string& name, Vec3 albedo, BrdfModel model, float roughness, float metallic);
BrdfModel parse_brdf_model(const std::string& name);
NprStyle parse_npr_style(const std::string& name);
const char* npr_style_name(NprStyle style);
XToonDetailMode parse_xtoon_detail_mode(const std::string& name);
const char* xtoon_detail_mode_name(XToonDetailMode mode);

} // namespace lt

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
    std::shared_ptr<Texture> albedo_texture;

    Material() = default;
    Material(std::string name_, Vec3 albedo_) : name(std::move(name_)), albedo(albedo_) {}
    virtual ~Material() = default;

    virtual BrdfModel model() const = 0;
    virtual const char* model_name() const = 0;
    Vec3 base_color(Vec2 uv) const;
    virtual Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const = 0;
    virtual float pdf(Vec3 n, Vec3 wo, Vec3 wi) const = 0;
    virtual MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const = 0;
    virtual std::shared_ptr<Material> clone() const = 0;
};

class LambertianMaterial final : public Material {
public:
    using Material::Material;

    BrdfModel model() const override { return BrdfModel::Lambertian; }
    const char* model_name() const override { return "lambertian"; }
    Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    float pdf(Vec3 n, Vec3 wo, Vec3 wi) const override;
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<LambertianMaterial>(*this); }
};

class PrincipledMaterial final : public Material {
public:
    float roughness = 0.5f;
    float metallic = 0.0f;

    PrincipledMaterial() = default;
    PrincipledMaterial(std::string name_, Vec3 albedo_, float roughness_, float metallic_)
        : Material(std::move(name_), albedo_), roughness(roughness_), metallic(metallic_) {}

    BrdfModel model() const override { return BrdfModel::Principled; }
    const char* model_name() const override { return "principled"; }
    Vec3 evaluate(Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) const override;
    float pdf(Vec3 n, Vec3 wo, Vec3 wi) const override;
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<PrincipledMaterial>(*this); }
};

class MirrorMaterial final : public Material {
public:
    using Material::Material;

    BrdfModel model() const override { return BrdfModel::Mirror; }
    const char* model_name() const override { return "mirror"; }
    Vec3 evaluate(Vec3, Vec3, Vec3, Vec2) const override { return {}; }
    float pdf(Vec3, Vec3, Vec3) const override { return 0.0f; }
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<MirrorMaterial>(*this); }
};

class DielectricMaterial final : public Material {
public:
    float ior = 1.5f;

    DielectricMaterial() = default;
    DielectricMaterial(std::string name_, Vec3 albedo_, float ior_)
        : Material(std::move(name_), albedo_), ior(ior_) {}

    BrdfModel model() const override { return BrdfModel::Dielectric; }
    const char* model_name() const override { return "dielectric"; }
    Vec3 evaluate(Vec3, Vec3, Vec3, Vec2) const override { return {}; }
    float pdf(Vec3, Vec3, Vec3) const override { return 0.0f; }
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<DielectricMaterial>(*this); }
};

class ConductorMaterial final : public Material {
public:
    using Material::Material;

    BrdfModel model() const override { return BrdfModel::Conductor; }
    const char* model_name() const override { return "conductor"; }
    Vec3 evaluate(Vec3, Vec3, Vec3, Vec2) const override { return {}; }
    float pdf(Vec3, Vec3, Vec3) const override { return 0.0f; }
    MaterialSample sample(Vec3 n, Vec3 wo, Vec2 uv, bool front_face, Rng& rng) const override;
    std::shared_ptr<Material> clone() const override { return std::make_shared<ConductorMaterial>(*this); }
};

std::shared_ptr<Material> make_material(const std::string& name, Vec3 albedo, BrdfModel model, float roughness, float metallic);
BrdfModel parse_brdf_model(const std::string& name);

} // namespace lt

enum class GpuRestirLightType : int {
    Triangle = 0,
    Directional = 1,
    Point = 2,
    Environment = 3,
};

struct GpuRestirLightSample {
    GpuRestirLightType type = GpuRestirLightType::Environment;
    int index = -1;
    Vec3 position_or_direction = {};
    Vec2 uv = {};
    int sampler_index = -1;
};

struct GpuRestirReservoir {
    GpuRestirLightSample sample;
    // Finalized reservoir weight (the selected sample's inverse PDF / UCW).
    float weight_sum = 0.0f;
    float selected_target = 0.0f;
    int M = 0;
    int valid = 0;
    int visibility = -1;
    int visibility_age = 0;
    int visibility_pixel = -1;
};

struct GpuRestirInitialReservoir {
    GpuRestirLightSample sample;
    float weight_sum = 0.0f;
    float selected_target = 0.0f;
    int M = 0;
    int valid = 0;
};

struct GpuRestirSurface {
    Vec3 position = {};
    Vec3 normal = {};
    Vec3 geo_normal = {};
    Vec3 wo = {};
    Vec2 uv = {};
    float depth = kInfinity;
    uint32_t object_id = 0u;
    int material = -1;
    int valid = 0;
};

struct GpuRestirVisibilityRay {
    Ray ray = {};
    float remaining = kInfinity;
    int sample_type = -1;
    int sample_index = -1;
    int path_index = -1;
    int valid = 0;
};

static constexpr int kRestirVisibilityGiTriangle = -2;
static constexpr int kRestirVisibilityGiSphere = -3;

__device__ bool restir_visibility_hit_is_target_gpu(const GpuRestirVisibilityRay& visibility,
    const GpuCompactHit& hit) {
    return (visibility.sample_type == static_cast<int>(GpuRestirLightType::Triangle) &&
            hit.triangle == visibility.sample_index) ||
        (visibility.sample_type == kRestirVisibilityGiTriangle && hit.triangle == visibility.sample_index) ||
        (visibility.sample_type == kRestirVisibilityGiSphere && hit.sphere == visibility.sample_index);
}

// One proposal per finite-light family, two HDRI proposals, and one BRDF
// proposal. The actual count is smaller when a family is absent.
static constexpr int kRestirInitialCandidates = 8;
static constexpr int kRestirEnvironmentCandidates = 1;
static constexpr int kRestirEnvironmentBrdfCandidates = 1;
static constexpr int kRestirTemporalM = 16;
static constexpr int kRestirMaxM = 32;
static constexpr int kRestirSpatialSamples = 1;
static constexpr int kRestirSpatialDisocclusionSamples = 2;
static constexpr int kRestirSpatialTargetHistory = 8;
static constexpr float kRestirSpatialRadius = 32.0f;
static constexpr float kRestirBoilingStrength = 0.20f;
static constexpr int kRestirFinalVisibilityMaxAge = 2;
static constexpr int kRestirFinalVisibilityMaxDistance = 1;
static constexpr int kRestirNeighborOffsetCount = 8192;

__device__ Vec3 restir_geometric_normal_gpu(const GpuScene& scene, const GpuHit& hit) {
    Vec3 normal = hit.normal;
    if (hit.triangle >= 0 && hit.triangle < scene.triangle_count) {
        normal = scene.triangles[hit.triangle].normal;
    } else if (hit.sphere >= 0 && hit.sphere < scene.sphere_count) {
        normal = dnormalize(sub(hit.position, scene.spheres[hit.sphere].center));
    }
    if (!(ddot(normal, normal) > 0.0f)) normal = hit.normal;
    normal = dnormalize(normal);
    return ddot(normal, hit.normal) >= 0.0f ? normal : mul(normal, -1.0f);
}

struct GpuRestirSpatialState {
    int start_index = 0;
    int sample_count = 0;
    int selected_neighbor = -1;
    int carried_m[kRestirSpatialDisocclusionSamples] = {};
};

struct GpuRestirTemporalState {
    int previous_pixel = -1;
    int previous_m = 0;
    int selected_previous = 0;
    uint32_t correction_rng = 0;
};

__device__ float restir_luminance_gpu(Vec3 color) {
    return fmaxf(0.0f, color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f);
}

__device__ uint32_t restir_rng_seed_gpu(uint32_t path_rng, uint32_t stream) {
    // Keep ReSTIR dimensions independent from path continuation and alpha-test dimensions.
    return hash_u32(path_rng ^ stream);
}

__device__ void restir_neighbor_offset_gpu(const Vec2* neighbor_offsets, int index, int& ox, int& oy) {
    const Vec2 offset = neighbor_offsets[index & (kRestirNeighborOffsetCount - 1)];
    ox = static_cast<int>(roundf(offset.x * kRestirSpatialRadius));
    oy = static_cast<int>(roundf(offset.y * kRestirSpatialRadius));
}

__device__ bool restir_direct_enabled_gpu(const RenderSettings& settings) {
    return settings.cuda_restir_di && settings.cuda_wavefront &&
        settings.sampling_mode != PathSamplingMode::Unidirectional;
}

__device__ bool restir_has_environment_light_gpu(const GpuScene& scene) {
    return scene.environment_strength > 0.0f &&
        (scene.environment_texture >= 0 || max_channel_gpu(scene.environment_color) > 0.0f);
}

__device__ Vec3 restir_environment_local_to_world_gpu(const GpuScene& scene, Vec3 local) {
    return dnormalize(add(add(mul(scene.environment_light_from_world_x, local.x),
        mul(scene.environment_light_from_world_y, local.y)),
        mul(scene.environment_light_from_world_z, local.z)));
}

__device__ Vec3 restir_equal_area_square_to_sphere_gpu(Vec2 uv) {
    const float u = dclamp(uv.x * 2.0f - 1.0f, -1.0f, 1.0f);
    const float v = dclamp(uv.y * 2.0f - 1.0f, -1.0f, 1.0f);
    const float up = fabsf(u);
    const float vp = fabsf(v);
    const float sum = up + vp;
    const float r = sum <= 1.0f ? sum : 2.0f - sum;
    if (r <= 1.0e-6f) {
        return {0.0f, 0.0f, 1.0f};
    }
    const float z = sum <= 1.0f ? 1.0f - r * r : r * r - 1.0f;
    const float a = dclamp((vp - up) / r, -1.0f, 1.0f);
    const float phi = (a + 1.0f) * kPi * 0.25f;
    const float radial = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return dnormalize({u < 0.0f ? -cosf(phi) * radial : cosf(phi) * radial,
        v < 0.0f ? -sinf(phi) * radial : sinf(phi) * radial, z});
}

__device__ Vec3 restir_environment_direction_from_uv_gpu(const GpuScene& scene, Vec2 uv) {
    Vec3 local;
    if (scene.environment_mapping == 1) {
        local = restir_equal_area_square_to_sphere_gpu(uv);
    } else {
        const float phi = (uv.x - 0.5f) * (2.0f * kPi);
        const float theta = (1.0f - uv.y) * kPi;
        const float sin_theta = sinf(theta);
        local = {cosf(phi) * sin_theta, cosf(theta), sinf(phi) * sin_theta};
    }
    return restir_environment_local_to_world_gpu(scene, local);
}

__device__ float restir_environment_pdf_gpu(const GpuScene& scene, Vec2 uv, int texel) {
    const GpuEnvironmentSampler sampler = scene.environment_sampler;
    if (sampler.texel_count <= 0 || sampler.pmf == nullptr || texel < 0 || texel >= sampler.texel_count) {
        return 1.0f / (4.0f * kPi);
    }
    const float pdf_uv = sampler.pmf[texel] * static_cast<float>(sampler.texel_count);
    if (scene.environment_mapping == 1) {
        return pdf_uv / (4.0f * kPi);
    }
    const float theta = (1.0f - uv.y) * kPi;
    return pdf_uv / fmaxf(1.0e-6f, 2.0f * kPi * kPi * sinf(theta));
}

__device__ bool restir_sample_environment_gpu(const GpuScene& scene, uint32_t& rng, GpuRestirLightSample& sample, float& pdf) {
    const GpuEnvironmentSampler sampler = scene.environment_sampler;
    Vec2 uv;
    int texel = -1;
    if (sampler.texel_count > 0 && sampler.width > 0 && sampler.height > 0 &&
        sampler.alias_probability != nullptr && sampler.alias_index != nullptr) {
        const float scaled = rng_float(rng) * static_cast<float>(sampler.texel_count);
        const int column = iclamp_gpu(static_cast<int>(scaled), 0, sampler.texel_count - 1);
        texel = rng_float(rng) < sampler.alias_probability[column] ? column : sampler.alias_index[column];
        const int x = texel % sampler.width;
        const int y = texel / sampler.width;
        uv = {(static_cast<float>(x) + rng_float(rng)) / static_cast<float>(sampler.width),
            (static_cast<float>(y) + rng_float(rng)) / static_cast<float>(sampler.height)};
    } else {
        if (scene.environment_mapping == 1) {
            // The equal-area map has a constant solid-angle Jacobian.
            uv = {rng_float(rng), rng_float(rng)};
        } else {
            const float y = 1.0f - 2.0f * rng_float(rng);
            uv = {rng_float(rng), 1.0f - acosf(dclamp(y, -1.0f, 1.0f)) / kPi};
        }
    }
    sample.type = GpuRestirLightType::Environment;
    sample.index = texel;
    sample.uv = uv;
    sample.position_or_direction = restir_environment_direction_from_uv_gpu(scene, uv);
    pdf = restir_environment_pdf_gpu(scene, uv, texel);
    return pdf > 0.0f && isfinite(pdf);
}

__device__ Vec2 restir_environment_uv_from_direction_gpu(const GpuScene& scene, Vec3 direction) {
    const Vec3 local = dnormalize({
        ddot(direction, scene.environment_light_from_world_x),
        ddot(direction, scene.environment_light_from_world_y),
        ddot(direction, scene.environment_light_from_world_z),
    });
    if (scene.environment_mapping == 1) {
        return equal_area_sphere_to_square_gpu(local);
    }
    return {atan2f(local.z, local.x) / (2.0f * kPi) + 0.5f,
        1.0f - acosf(dclamp(local.y, -1.0f, 1.0f)) / kPi};
}

__device__ int restir_environment_texel_from_uv_gpu(const GpuScene& scene, Vec2 uv) {
    const GpuEnvironmentSampler sampler = scene.environment_sampler;
    if (sampler.texel_count <= 0 || sampler.width <= 0 || sampler.height <= 0) return -1;
    const int x = iclamp_gpu(static_cast<int>(uv.x * static_cast<float>(sampler.width)), 0, sampler.width - 1);
    const int y = iclamp_gpu(static_cast<int>(uv.y * static_cast<float>(sampler.height)), 0, sampler.height - 1);
    return y * sampler.width + x;
}

__device__ float restir_environment_mis_proposal_pdf_gpu(const GpuScene& scene, const GpuHit& hit,
    const GpuMaterial& material, Vec3 wo, const GpuRestirLightSample& sample,
    int environment_sample_count, int brdf_sample_count, int total_sample_count) {
    const float environment_pdf = restir_environment_pdf_gpu(scene, sample.uv, sample.index);
    const float brdf_pdf = material_pdf_gpu(scene, material, hit.normal, wo, sample.position_or_direction, hit.uv);
    return total_sample_count > 0 ?
        (environment_pdf * static_cast<float>(environment_sample_count) +
            brdf_pdf * static_cast<float>(brdf_sample_count)) / static_cast<float>(total_sample_count) : 0.0f;
}

__device__ bool restir_sample_triangle_gpu(const GpuScene& scene, uint32_t& rng,
    GpuRestirLightSample& sample, float& discrete_pdf) {
    if (scene.light_count <= 0 || scene.light_indices == nullptr) return false;
    const GpuLightSampler sampler = scene.restir_light_sampler;
    int list_index = 0;
    if (sampler.light_count == scene.light_count && sampler.pmf != nullptr &&
        sampler.alias_probability != nullptr && sampler.alias_index != nullptr) {
        const float scaled = rng_float(rng) * static_cast<float>(sampler.light_count);
        const int column = iclamp_gpu(static_cast<int>(scaled), 0, sampler.light_count - 1);
        list_index = rng_float(rng) < sampler.alias_probability[column] ? column : sampler.alias_index[column];
        discrete_pdf = sampler.pmf[list_index];
    } else {
        list_index = iclamp_gpu(static_cast<int>(rng_float(rng) * static_cast<float>(scene.light_count)), 0, scene.light_count - 1);
        discrete_pdf = 1.0f / static_cast<float>(scene.light_count);
    }
    const int light_index = scene.light_indices[list_index];
    if (light_index < 0 || light_index >= scene.triangle_count || !(discrete_pdf > 0.0f)) return false;
    const GpuTriangle light = scene.triangles[light_index];
    sample = {};
    sample.type = GpuRestirLightType::Triangle;
    sample.index = light_index;
    sample.sampler_index = list_index;
    sample.uv = sample_triangle_uv_gpu(light, rng, sample.position_or_direction);
    return true;
}

__device__ float restir_triangle_proposal_pdf_gpu(const GpuScene& scene, const GpuHit& hit,
    const GpuRestirLightSample& sample) {
    if (sample.index < 0 || sample.index >= scene.triangle_count) return 0.0f;
    const GpuLightSampler sampler = scene.restir_light_sampler;
    const float discrete_pdf = sampler.light_count == scene.light_count && sampler.pmf != nullptr &&
        sample.sampler_index >= 0 && sample.sampler_index < sampler.light_count
        ? sampler.pmf[sample.sampler_index]
        : (scene.light_count > 0 ? 1.0f / static_cast<float>(scene.light_count) : 0.0f);
    const GpuTriangle light = scene.triangles[sample.index];
    return light_pdf_solid_angle_gpu(light, scene.materials[light.material], hit.position,
        sample.position_or_direction, discrete_pdf);
}

__device__ bool restir_sample_directional_gpu(const GpuScene& scene, uint32_t& rng,
    GpuRestirLightSample& sample) {
    if (scene.directional_light_count <= 0) return false;
    sample = {};
    sample.type = GpuRestirLightType::Directional;
    sample.index = iclamp_gpu(static_cast<int>(rng_float(rng) * static_cast<float>(scene.directional_light_count)),
        0, scene.directional_light_count - 1);
    sample.position_or_direction = dnormalize(scene.directional_lights[sample.index].direction);
    return true;
}

__device__ bool restir_sample_point_gpu(const GpuScene& scene, uint32_t& rng,
    GpuRestirLightSample& sample) {
    if (scene.point_light_count <= 0) return false;
    sample = {};
    sample.type = GpuRestirLightType::Point;
    sample.index = iclamp_gpu(static_cast<int>(rng_float(rng) * static_cast<float>(scene.point_light_count)),
        0, scene.point_light_count - 1);
    sample.position_or_direction = scene.point_lights[sample.index].position;
    return true;
}

__device__ float restir_surface_ndotl_gpu(const GpuMaterial& material, const GpuHit& hit, Vec3 direction, float& raw) {
    raw = ddot(hit.normal, direction);
    const bool diffuse_transmission = material.brdf_model == static_cast<int>(BrdfModel::DiffuseTransmission);
    return (material.double_sided || diffuse_transmission) ? fabsf(raw) : fmaxf(0.0f, raw);
}

__device__ Vec3 restir_evaluate_environment_sample_gpu(const GpuScene& scene, const GpuHit& hit,
    const GpuMaterial& material, Vec3 wo, const RenderSettings& settings,
    const GpuRestirLightSample& sample, Vec3& direction, float& distance) {
    direction = sample.position_or_direction;
    distance = kInfinity;
    float ndotl_raw = 0.0f;
    const float ndotl = restir_surface_ndotl_gpu(material, hit, direction, ndotl_raw);
    if (ndotl <= 0.0f) return {};
    const Vec3 radiance = mul(mul(evaluate_brdf_gpu(scene, material, hit.normal, wo, direction, hit.uv),
        environment_radiance_gpu(scene, direction, settings)), ndotl);
    return finite_vec_gpu(radiance) ? radiance : Vec3{};
}

__device__ Vec3 restir_evaluate_triangle_sample_gpu(const GpuScene& scene, const GpuHit& hit,
    const GpuMaterial& material, Vec3 wo, const GpuRestirLightSample& sample,
    const RenderSettings& settings, Vec3& direction, float& distance) {
    direction = {};
    distance = kInfinity;
    if (sample.index < 0 || sample.index >= scene.triangle_count) return {};
    const GpuTriangle light = scene.triangles[sample.index];
    const Vec3 to_light = sub(sample.position_or_direction, hit.position);
    const float dist2 = ddot(to_light, to_light);
    if (dist2 <= 1.0e-8f) return {};
    distance = sqrtf(dist2);
    direction = divv(to_light, distance);
    float ndotl_raw = 0.0f;
    const float ndotl = restir_surface_ndotl_gpu(material, hit, direction, ndotl_raw);
    const GpuMaterial light_material = scene.materials[light.material];
    const float ldot_raw = ddot(mul(light.normal, -1.0f), direction);
    const bool material_double_sided = !has_light_emission_gpu(light.emission) && light_material.double_sided &&
        has_light_emission_gpu(material_emission_gpu(scene, light_material, sample.uv, settings));
    const float ldot = light.light_double_sided != 0 || material_double_sided
        ? fabsf(ldot_raw) : fmaxf(0.0f, ldot_raw);
    if (ndotl <= 0.0f || ldot <= 0.0f) return {};
    const Vec3 emission = emitted_radiance_gpu(light, light_material, light.emission,
        material_emission_gpu(scene, light_material, sample.uv, settings), light.light_double_sided != 0, direction);
    const Vec3 radiance = mul(mul(evaluate_brdf_gpu(
        scene, material, hit.normal, wo, direction, hit.uv), emission), ndotl);
    return finite_vec_gpu(radiance) ? radiance : Vec3{};
}

__device__ Vec3 restir_evaluate_directional_sample_gpu(const GpuScene& scene, const GpuHit& hit,
    const GpuMaterial& material, Vec3 wo, const GpuRestirLightSample& sample,
    const RenderSettings& settings, Vec3& direction, float& distance) {
    direction = {};
    distance = kInfinity;
    if (sample.index < 0 || sample.index >= scene.directional_light_count) return {};
    const GpuDirectionalLight light = scene.directional_lights[sample.index];
    if (light.intensity <= 0.0f || ddot(light.direction, light.direction) <= 0.0f) return {};
    direction = dnormalize(light.direction);
    float ndotl_raw = 0.0f;
    const float ndotl = restir_surface_ndotl_gpu(material, hit, direction, ndotl_raw);
    if (ndotl <= 0.0f) return {};
    const Vec3 radiance = mul(mul(evaluate_brdf_gpu(scene, material, hit.normal, wo, direction, hit.uv),
        mul(light.color, light.intensity)), ndotl);
    return finite_vec_gpu(radiance) ? radiance : Vec3{};
}

__device__ Vec3 restir_evaluate_point_sample_gpu(const GpuScene& scene, const GpuHit& hit,
    const GpuMaterial& material, Vec3 wo, const GpuRestirLightSample& sample,
    const RenderSettings& settings, Vec3& direction, float& distance) {
    direction = {};
    distance = kInfinity;
    if (sample.index < 0 || sample.index >= scene.point_light_count) return {};
    const GpuPointLight light = scene.point_lights[sample.index];
    if (light.intensity <= 0.0f) return {};
    const Vec3 to_light = sub(light.position, hit.position);
    const float dist2 = ddot(to_light, to_light);
    if (dist2 <= 1.0e-8f) return {};
    distance = sqrtf(dist2);
    direction = divv(to_light, distance);
    float ndotl_raw = 0.0f;
    const float ndotl = restir_surface_ndotl_gpu(material, hit, direction, ndotl_raw);
    if (ndotl <= 0.0f) return {};
    const Vec3 radiance = mul(mul(evaluate_brdf_gpu(scene, material, hit.normal, wo, direction, hit.uv),
        mul(light.color, light.intensity / fmaxf(0.001f, dist2))), ndotl);
    return finite_vec_gpu(radiance) ? radiance : Vec3{};
}

__device__ Vec3 restir_evaluate_sample_gpu(const GpuScene& scene, const GpuHit& hit, const GpuMaterial& material,
    Vec3 wo, const RenderSettings& settings, const GpuRestirLightSample& sample, Vec3& direction, float& distance) {
    if (sample.type == GpuRestirLightType::Triangle) {
        return restir_evaluate_triangle_sample_gpu(scene, hit, material, wo, sample, settings, direction, distance);
    } else if (sample.type == GpuRestirLightType::Directional) {
        return restir_evaluate_directional_sample_gpu(scene, hit, material, wo, sample, settings, direction, distance);
    } else if (sample.type == GpuRestirLightType::Point) {
        return restir_evaluate_point_sample_gpu(scene, hit, material, wo, sample, settings, direction, distance);
    } else {
        return restir_evaluate_environment_sample_gpu(scene, hit, material, wo, settings,
            sample, direction, distance);
    }
}

__device__ GpuHit restir_surface_hit_gpu(const GpuRestirSurface& surface) {
    GpuHit hit{};
    hit.position = surface.position;
    hit.normal = surface.normal;
    hit.uv = surface.uv;
    return hit;
}

__device__ Vec3 restir_evaluate_surface_sample_gpu(const GpuScene& scene, const GpuRestirSurface& surface,
    const RenderSettings& settings, const GpuRestirLightSample& sample, Vec3& direction, float& distance) {
    if (!surface.valid || surface.material < 0 || surface.material >= scene.material_count) {
        direction = {};
        distance = kInfinity;
        return {};
    }
    const GpuHit hit = restir_surface_hit_gpu(surface);
    return restir_evaluate_sample_gpu(scene, hit, scene.materials[surface.material], surface.wo,
        settings, sample, direction, distance);
}

__device__ bool restir_sample_visible_gpu(const GpuScene& scene, const GpuHit& hit, const GpuMaterial& material,
    const GpuRestirLightSample& sample, Vec3 direction, float distance, uint32_t& rng) {
    float ndotl_raw = 0.0f;
    restir_surface_ndotl_gpu(material, hit, direction, ndotl_raw);
    Ray shadow_ray{add(hit.position, mul(hit.normal, 0.002f * (ndotl_raw >= 0.0f ? 1.0f : -1.0f))), direction};
    float remaining = isfinite(distance) ? distance - 0.01f : kInfinity;
    for (int step = 0; step < 8; ++step) {
        GpuHit shadow_hit;
        shadow_hit.t = remaining;
        if (!intersect_gpu(scene, shadow_ray, shadow_hit)) return true;
        if (sample.type == GpuRestirLightType::Triangle && shadow_hit.triangle == sample.index) return true;
        const GpuMaterial blocker = scene.materials[shadow_hit.material];
        if (!material_visible_gpu(scene, blocker, shadow_hit.uv, rng) ||
            blocker.brdf_model == static_cast<int>(BrdfModel::Dielectric) ||
            material_transmission_gpu(scene, blocker, shadow_hit.uv) > 0.5f) {
            shadow_ray = {add(shadow_hit.position, mul(direction, 0.002f)), direction};
            if (isfinite(remaining)) {
                remaining -= shadow_hit.t + 0.002f;
                if (remaining <= 0.001f) return true;
            }
            continue;
        }
        return false;
    }
    return true;
}

__device__ bool restir_cached_visibility_valid_gpu(const GpuRestirReservoir& reservoir,
    RenderSettings settings, int pixel) {
    if (!settings.cuda_restir_final_visibility_reuse || reservoir.visibility < 0 ||
        reservoir.visibility_age > kRestirFinalVisibilityMaxAge || reservoir.visibility_pixel < 0) {
        return false;
    }
    const int x = pixel % settings.width;
    const int y = pixel / settings.width;
    const int previous_x = reservoir.visibility_pixel % settings.width;
    const int previous_y = reservoir.visibility_pixel / settings.width;
    return abs(x - previous_x) <= kRestirFinalVisibilityMaxDistance &&
        abs(y - previous_y) <= kRestirFinalVisibilityMaxDistance;
}

__device__ float restir_source_target_gpu(const GpuScene& scene, const GpuRestirSurface& surface,
    const RenderSettings& settings, const GpuRestirLightSample& sample,
    bool test_conservative_visibility, uint32_t& rng) {
    Vec3 direction;
    float distance = 0.0f;
    const Vec3 contribution = restir_evaluate_surface_sample_gpu(scene, surface, settings, sample, direction, distance);
    float target = restir_luminance_gpu(contribution);
    if (target <= 0.0f || !test_conservative_visibility ||
        surface.material < 0 || surface.material >= scene.material_count) {
        return target;
    }
    const GpuHit hit = restir_surface_hit_gpu(surface);
    return restir_sample_visible_gpu(scene, hit, scene.materials[surface.material], sample, direction, distance, rng)
        ? target : 0.0f;
}

__device__ void restir_clear_gpu(GpuRestirReservoir& reservoir) {
    reservoir = {};
}

__device__ void restir_clear_initial_gpu(GpuRestirInitialReservoir& reservoir) {
    reservoir = {};
}

__device__ void restir_add_initial_gpu(GpuRestirInitialReservoir& reservoir, const GpuRestirLightSample& sample,
    float target, float proposal_pdf, uint32_t& rng) {
    if (!(proposal_pdf > 0.0f) || !isfinite(proposal_pdf) || reservoir.M >= kRestirInitialCandidates) return;
    ++reservoir.M;
    if (!(target > 0.0f) || !isfinite(target)) return;
    const float weight = target / proposal_pdf;
    const float next_sum = reservoir.weight_sum + weight;
    if (rng_float(rng) * next_sum < weight) {
        reservoir.sample = sample;
        reservoir.selected_target = target;
        reservoir.valid = 1;
    }
    reservoir.weight_sum = next_sum;
}

__device__ void restir_finalize_initial_gpu(GpuRestirInitialReservoir& reservoir,
    float numerator, float denominator) {
    const float normalization = reservoir.selected_target * denominator;
    reservoir.weight_sum = normalization > 0.0f && numerator > 0.0f
        ? reservoir.weight_sum * numerator / normalization : 0.0f;
    reservoir.valid = reservoir.weight_sum > 0.0f && isfinite(reservoir.weight_sum) ? reservoir.valid : 0;
}

__device__ void restir_finalize_gpu(GpuRestirReservoir& reservoir, float numerator, float denominator) {
    const float normalization = reservoir.selected_target * denominator;
    reservoir.weight_sum = normalization > 0.0f && numerator > 0.0f
        ? reservoir.weight_sum * numerator / normalization : 0.0f;
    reservoir.valid = reservoir.weight_sum > 0.0f && isfinite(reservoir.weight_sum) ? reservoir.valid : 0;
}

__device__ void restir_finalize_initial_strategy_gpu(GpuRestirInitialReservoir& reservoir,
    int requested_samples, float normalization_denominator) {
    if (requested_samples <= 0) return;
    restir_finalize_initial_gpu(reservoir, 1.0f, normalization_denominator);
    // RTXDI treats a completed initial strategy as one estimator when the
    // strategy reservoirs are combined, even if all of its targets were zero.
    reservoir.M = 1;
}

__device__ bool restir_stream_initial_strategy_gpu(GpuRestirReservoir& reservoir,
    const GpuRestirInitialReservoir& source, int max_m, uint32_t& rng) {
    if (source.M <= 0 || reservoir.M >= max_m) return false;
    ++reservoir.M;
    if (!source.valid || !(source.weight_sum > 0.0f) || !(source.selected_target > 0.0f)) return false;
    const float weight = source.selected_target * source.weight_sum;
    if (!(weight > 0.0f) || !isfinite(weight)) return false;
    const float next_sum = reservoir.weight_sum + weight;
    bool selected = false;
    if (rng_float(rng) * next_sum < weight) {
        reservoir.sample = source.sample;
        reservoir.selected_target = source.selected_target;
        reservoir.valid = 1;
        reservoir.visibility = -1;
        reservoir.visibility_age = 0;
        reservoir.visibility_pixel = -1;
        selected = true;
    }
    reservoir.weight_sum = next_sum;
    return selected;
}

__device__ bool restir_stream_reservoir_gpu(GpuRestirReservoir& reservoir, const GpuRestirReservoir& source,
    float target_at_receiver, int max_m, int source_m_limit, uint32_t& rng) {
    if (source.M <= 0 || reservoir.M >= max_m) return false;
    const int source_m = source.M < source_m_limit ? source.M : source_m_limit;
    const int carried_m = source_m < (max_m - reservoir.M) ? source_m : (max_m - reservoir.M);
    if (carried_m <= 0) return false;
    // RTXDI_InternalSimpleResample always carries M, including zero-weight or
    // invalid sources. Dropping it makes future weightSum * M reconstruction invalid.
    reservoir.M += carried_m;
    if (!source.valid || !(source.weight_sum > 0.0f) || !(target_at_receiver > 0.0f)) return false;
    const float weight = target_at_receiver * source.weight_sum * static_cast<float>(carried_m);
    if (!(weight > 0.0f) || !isfinite(weight)) return false;
    const float next_sum = reservoir.weight_sum + weight;
    bool selected = false;
    if (rng_float(rng) * next_sum < weight) {
        reservoir.sample = source.sample;
        reservoir.selected_target = target_at_receiver;
        reservoir.valid = 1;
        reservoir.visibility = source.visibility;
        reservoir.visibility_age = source.visibility >= 0 ? source.visibility_age + 1 : 0;
        reservoir.visibility_pixel = source.visibility_pixel;
        selected = true;
    }
    reservoir.weight_sum = next_sum;
    return selected;
}

__device__ bool restir_project_to_camera_gpu(const Camera& camera, Vec3 position, int width, int height,
    float& pixel_x, float& pixel_y, float& depth) {
    const Vec3 forward = dnormalize(sub(camera.target, camera.position));
    const Vec3 right = mul(dnormalize(dcross(forward, camera.up)), camera.right_sign < 0.0f ? -1.0f : 1.0f);
    const Vec3 up = dnormalize(dcross(right, forward));
    const Vec3 relative = sub(position, camera.position);
    depth = ddot(relative, forward);
    if (!(depth > 1.0e-5f)) return false;
    const float half_height = tanf(camera.fov_degrees * kPi / 360.0f);
    const float half_width = half_height * static_cast<float>(width) / static_cast<float>(height);
    const float nx = ddot(relative, right) / (depth * half_width);
    const float ny = ddot(relative, up) / (depth * half_height);
    pixel_x = (nx * 0.5f + 0.5f) * static_cast<float>(width) - 0.5f;
    pixel_y = (0.5f - ny * 0.5f) * static_cast<float>(height) - 0.5f;
    return pixel_x >= -0.5f && pixel_x < static_cast<float>(width) - 0.5f &&
        pixel_y >= -0.5f && pixel_y < static_cast<float>(height) - 0.5f;
}

__device__ bool restir_surfaces_compatible_gpu(const GpuRestirSurface& current, const GpuRestirSurface& other) {
    return current.valid && other.valid && current.object_id == other.object_id && current.material == other.material &&
        ddot(current.normal, other.normal) > 0.90f &&
        fabsf(current.depth - other.depth) <= fmaxf(0.02f, current.depth * 0.02f);
}

__device__ bool restir_temporal_surfaces_compatible_gpu(const GpuRestirSurface& current,
    const GpuRestirSurface& other, float expected_history_depth) {
    return current.valid && other.valid && current.object_id == other.object_id && current.material == other.material &&
        ddot(current.normal, other.normal) > 0.90f &&
        fabsf(other.depth - expected_history_depth) <= fmaxf(0.02f, expected_history_depth * 0.02f);
}

__global__ void restir_clear_surfaces_kernel(GpuRestirSurface* surfaces, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel < pixel_count) surfaces[pixel].valid = 0;
}

template <bool HasLocalLights, bool HasEnvironment>
__global__ void restir_initial_candidates_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuHit* hits, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, GpuRestirReservoir* initial,
    GpuRestirSurface* current_surfaces) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const GpuScene& scene = *scene_ptr;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;
    const GpuHit& hit = hits[path_index];
    const GpuMaterial& material = scene.materials[hit.material];
    GpuRestirSurface surface;
    surface.position = hit.position;
    surface.normal = hit.normal;
    surface.geo_normal = restir_geometric_normal_gpu(scene, hit);
    surface.wo = mul(path.ray.direction, -1.0f);
    surface.uv = hit.uv;
    const Vec3 forward = dnormalize(sub(scene.camera.target, scene.camera.position));
    surface.depth = fmaxf(0.0f, ddot(sub(hit.position, scene.camera.position), forward));
    surface.object_id = primary_object_id_gpu(hit);
    surface.material = hit.material;
    surface.valid = 1;
    current_surfaces[path.pixel] = surface;
    GpuRestirReservoir reservoir;
    restir_clear_gpu(reservoir);
    GpuRestirInitialReservoir strategy_reservoir;
    uint32_t restir_rng = restir_rng_seed_gpu(path.rng, 0x68bc21ebu);
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    Vec3 brdf_direction = {};
    bool brdf_sample_valid = false;
    if constexpr (HasEnvironment) {
        const GpuMaterialSample brdf_sample =
            sample_material_gpu(scene, material, hit.normal, wo, hit.uv, hit.front_face, restir_rng);
        brdf_direction = brdf_sample.direction;
        brdf_sample_valid = !brdf_sample.delta && brdf_sample.pdf > 0.0f && isfinite(brdf_sample.pdf);
    }
    int triangle_sample_count = HasLocalLights && scene.light_count > 0 ? 1 : 0;
    int directional_sample_count = HasLocalLights && scene.directional_light_count > 0 ? 1 : 0;
    int point_sample_count = HasLocalLights && scene.point_light_count > 0 ? 1 : 0;
    int environment_sample_count = HasEnvironment ? kRestirEnvironmentCandidates : 0;
    const int brdf_sample_count = brdf_sample_valid ? kRestirEnvironmentBrdfCandidates : 0;
    int total_sample_count = triangle_sample_count + directional_sample_count + point_sample_count +
        environment_sample_count + brdf_sample_count;
    // Keep RTXDI-style pool budgets independent. The eight-sample budget belongs
    // to finite local lights; absent local lights do not donate slots to the
    // environment or infinite-light pools.
    while (total_sample_count < kRestirInitialCandidates) {
        if (triangle_sample_count > 0) {
            ++triangle_sample_count;
        } else if (point_sample_count > 0) {
            ++point_sample_count;
        } else {
            break;
        }
        ++total_sample_count;
    }
    const int environment_mis_sample_count = environment_sample_count + brdf_sample_count;
    restir_clear_initial_gpu(strategy_reservoir);
    for (int i = 0; i < triangle_sample_count; ++i) {
        GpuRestirLightSample sample;
        float discrete_pdf = 0.0f;
        if (!restir_sample_triangle_gpu(scene, restir_rng, sample, discrete_pdf)) continue;
        Vec3 direction;
        float distance = 0.0f;
        const Vec3 contribution = restir_evaluate_triangle_sample_gpu(
            scene, hit, material, wo, sample, settings, direction, distance);
        const float target = restir_luminance_gpu(contribution);
        const float proposal = restir_triangle_proposal_pdf_gpu(scene, hit, sample);
        restir_add_initial_gpu(strategy_reservoir, sample, target, proposal, restir_rng);
    }
    restir_finalize_initial_strategy_gpu(strategy_reservoir, triangle_sample_count,
        static_cast<float>(triangle_sample_count));
    restir_stream_initial_strategy_gpu(reservoir, strategy_reservoir, kRestirInitialCandidates, restir_rng);

    restir_clear_initial_gpu(strategy_reservoir);
    for (int i = 0; i < directional_sample_count; ++i) {
        GpuRestirLightSample sample;
        if (!restir_sample_directional_gpu(scene, restir_rng, sample)) continue;
        Vec3 direction;
        float distance = 0.0f;
        const Vec3 contribution = restir_evaluate_directional_sample_gpu(
            scene, hit, material, wo, sample, settings, direction, distance);
        const float proposal = 1.0f / static_cast<float>(scene.directional_light_count);
        restir_add_initial_gpu(strategy_reservoir, sample, restir_luminance_gpu(contribution), proposal, restir_rng);
    }
    restir_finalize_initial_strategy_gpu(strategy_reservoir, directional_sample_count,
        static_cast<float>(directional_sample_count));
    restir_stream_initial_strategy_gpu(reservoir, strategy_reservoir, kRestirInitialCandidates, restir_rng);

    restir_clear_initial_gpu(strategy_reservoir);
    for (int i = 0; i < point_sample_count; ++i) {
        GpuRestirLightSample sample;
        if (!restir_sample_point_gpu(scene, restir_rng, sample)) continue;
        Vec3 direction;
        float distance = 0.0f;
        const Vec3 contribution = restir_evaluate_point_sample_gpu(
            scene, hit, material, wo, sample, settings, direction, distance);
        const float proposal = 1.0f / static_cast<float>(scene.point_light_count);
        restir_add_initial_gpu(strategy_reservoir, sample, restir_luminance_gpu(contribution), proposal, restir_rng);
    }
    restir_finalize_initial_strategy_gpu(strategy_reservoir, point_sample_count,
        static_cast<float>(point_sample_count));
    restir_stream_initial_strategy_gpu(reservoir, strategy_reservoir, kRestirInitialCandidates, restir_rng);

    restir_clear_initial_gpu(strategy_reservoir);
    for (int i = 0; i < environment_sample_count; ++i) {
        GpuRestirLightSample sample;
        float environment_pdf = 0.0f;
        if (!restir_sample_environment_gpu(scene, restir_rng, sample, environment_pdf)) continue;
        Vec3 direction;
        float distance = 0.0f;
        const Vec3 contribution = restir_evaluate_environment_sample_gpu(
            scene, hit, material, wo, settings, sample, direction, distance);
        const float proposal = restir_environment_mis_proposal_pdf_gpu(scene, hit, material, wo, sample,
            environment_sample_count, brdf_sample_count, environment_mis_sample_count);
        restir_add_initial_gpu(strategy_reservoir, sample, restir_luminance_gpu(contribution), proposal, restir_rng);
    }
    restir_finalize_initial_strategy_gpu(strategy_reservoir, environment_sample_count,
        static_cast<float>(environment_mis_sample_count));
    restir_stream_initial_strategy_gpu(reservoir, strategy_reservoir, kRestirInitialCandidates, restir_rng);

    restir_clear_initial_gpu(strategy_reservoir);
    if (brdf_sample_count > 0) {
        GpuRestirLightSample sample;
        sample.type = GpuRestirLightType::Environment;
        sample.position_or_direction = brdf_direction;
        sample.uv = restir_environment_uv_from_direction_gpu(scene, brdf_direction);
        sample.index = restir_environment_texel_from_uv_gpu(scene, sample.uv);
        Vec3 direction;
        float distance = 0.0f;
        const Vec3 contribution = restir_evaluate_environment_sample_gpu(
            scene, hit, material, wo, settings, sample, direction, distance);
        const float proposal = restir_environment_mis_proposal_pdf_gpu(scene, hit, material, wo, sample,
            environment_sample_count, brdf_sample_count, environment_mis_sample_count);
        restir_add_initial_gpu(strategy_reservoir, sample, restir_luminance_gpu(contribution), proposal, restir_rng);
    }
    restir_finalize_initial_strategy_gpu(strategy_reservoir, brdf_sample_count,
        static_cast<float>(environment_mis_sample_count));
    restir_stream_initial_strategy_gpu(reservoir, strategy_reservoir, kRestirInitialCandidates, restir_rng);

    // Match RTXDI_SampleLightsForSurface: combine completed strategy
    // reservoirs, finalize that combination, then collapse it to M=1.
    restir_finalize_gpu(reservoir, 1.0f, 1.0f);
    reservoir.M = 1;
    initial[path.pixel] = reservoir;
}

__global__ void restir_temporal_resample_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuHit* hits, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, const GpuRestirReservoir* initial,
    const GpuRestirReservoir* history, const GpuRestirSurface* current_surfaces,
    const GpuRestirSurface* history_surfaces, GpuRestirReservoir* temporal,
    GpuRestirTemporalState* temporal_states, Camera history_camera, bool history_valid) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const GpuScene& scene = *scene_ptr;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;
    const GpuRestirReservoir canonical = initial[path.pixel];
    GpuRestirReservoir reservoir;
    restir_clear_gpu(reservoir);
    uint32_t restir_rng = restir_rng_seed_gpu(path.rng, 0x02e5be93u);
    const GpuRestirSurface current = current_surfaces[path.pixel];
    restir_stream_reservoir_gpu(reservoir, canonical, canonical.selected_target,
        kRestirMaxM, canonical.M, restir_rng);
    GpuRestirTemporalState state;
    if (history_valid) {
        float previous_x = 0.0f;
        float previous_y = 0.0f;
        float previous_depth = 0.0f;
        if (restir_project_to_camera_gpu(history_camera, current.position, settings.width, settings.height,
                previous_x, previous_y, previous_depth)) {
            const int center_x = iclamp_gpu(static_cast<int>(previous_x + 0.5f), 0, settings.width - 1);
            const int center_y = iclamp_gpu(static_cast<int>(previous_y + 0.5f), 0, settings.height - 1);
            float best_score = kInfinity;
            // Reprojection is not guaranteed to land on the exact rasterized history pixel.
            // Scan the fixed 3x3 footprint and take the closest compatible surface.
            for (int oy = -1; oy <= 1; ++oy) {
                for (int ox = -1; ox <= 1; ++ox) {
                    const int px = center_x + ox;
                    const int py = center_y + oy;
                    if (px < 0 || px >= settings.width || py < 0 || py >= settings.height) continue;
                    const int candidate_pixel = py * settings.width + px;
                    const GpuRestirSurface candidate_surface = history_surfaces[candidate_pixel];
                    if (!restir_temporal_surfaces_compatible_gpu(current, candidate_surface, previous_depth)) continue;
                    const float dx = static_cast<float>(px) - previous_x;
                    const float dy = static_cast<float>(py) - previous_y;
                    const float depth_error = fabsf(candidate_surface.depth - previous_depth) /
                        fmaxf(0.01f, previous_depth);
                    const float score = dx * dx + dy * dy + 4.0f * depth_error;
                    if (score < best_score) {
                        best_score = score;
                        state.previous_pixel = candidate_pixel;
                    }
                }
            }
            if (state.previous_pixel >= 0) {
                Vec3 direction;
                float distance = 0.0f;
                const Vec3 contribution = restir_evaluate_surface_sample_gpu(scene, current, settings,
                    history[state.previous_pixel].sample, direction, distance);
                state.previous_m = history[state.previous_pixel].M < kRestirTemporalM
                    ? history[state.previous_pixel].M : kRestirTemporalM;
                state.selected_previous = restir_stream_reservoir_gpu(reservoir, history[state.previous_pixel],
                    restir_luminance_gpu(contribution), kRestirMaxM, state.previous_m, restir_rng) ? 1 : 0;
            }
        }
    }
    state.correction_rng = restir_rng;
    temporal_states[path.pixel] = state;
    temporal[path.pixel] = reservoir;
}

__global__ void restir_temporal_finalize_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, const GpuRestirReservoir* initial,
    const GpuRestirSurface* history_surfaces, const GpuRestirTemporalState* temporal_states,
    GpuRestirReservoir* temporal) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const GpuScene& scene = *scene_ptr;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;
    const int pixel = path.pixel;
    GpuRestirReservoir reservoir = temporal[pixel];
    const GpuRestirReservoir canonical = initial[pixel];
    const GpuRestirTemporalState state = temporal_states[pixel];
    if (reservoir.valid) {
        // state.targetPdf in RTXDI is always the selected sample's target at the receiver.
        // Ray-traced correction is only applied to the non-canonical source term.
        const float current_target = reservoir.selected_target;
        float previous_target = 0.0f;
        uint32_t restir_rng = state.correction_rng;
        if (state.previous_pixel >= 0 && state.previous_m > 0) {
            const GpuRestirSurface previous_surface = history_surfaces[state.previous_pixel];
            previous_target = restir_source_target_gpu(scene, previous_surface, settings, reservoir.sample,
                settings.cuda_restir_bias_correction == RestirBiasCorrection::RayTraced, restir_rng);
        }
        const float numerator = state.selected_previous != 0 ? previous_target : current_target;
        const float denominator = current_target * static_cast<float>(canonical.M) +
            previous_target * static_cast<float>(state.previous_m);
        restir_finalize_gpu(reservoir, numerator, denominator);
    }
    temporal[pixel] = reservoir;
}

__global__ void restir_boiling_filter_kernel(const GpuRestirSurface* surfaces,
    GpuRestirReservoir* reservoirs, int width, int height) {
    __shared__ float weights[256];
    __shared__ int nonzero[256];
    const int local = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const bool inside = x < width && y < height;
    const int pixel = y * width + x;
    const bool active = inside && surfaces[pixel].valid;
    const GpuRestirReservoir reservoir = active ? reservoirs[pixel] : GpuRestirReservoir{};
    const float weight = reservoir.valid && isfinite(reservoir.weight_sum)
        ? fmaxf(0.0f, reservoir.weight_sum) : 0.0f;
    weights[local] = weight;
    nonzero[local] = weight > 0.0f ? 1 : 0;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (local < stride) {
            weights[local] += weights[local + stride];
            nonzero[local] += nonzero[local + stride];
        }
        __syncthreads();
    }
    if (!active) return;
    const float average = nonzero[0] > 0 ? weights[0] / static_cast<float>(nonzero[0]) : 0.0f;
    const float threshold_multiplier = 10.0f / kRestirBoilingStrength - 9.0f;
    if (weight > average * threshold_multiplier) {
        reservoirs[pixel] = {};
    }
}

__global__ void restir_spatial_resample_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuHit* hits, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, const GpuRestirReservoir* temporal,
    const GpuRestirSurface* current_surfaces, const Vec2* neighbor_offsets, GpuRestirReservoir* output,
    GpuRestirSpatialState* spatial_states) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const GpuScene& scene = *scene_ptr;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;
    const GpuRestirReservoir canonical = temporal[path.pixel];
    GpuRestirReservoir reservoir;
    restir_clear_gpu(reservoir);
    const GpuRestirSurface current = current_surfaces[path.pixel];
    const int x = path.pixel % settings.width;
    const int y = path.pixel / settings.width;
    uint32_t restir_rng = restir_rng_seed_gpu(path.rng, 0xb5297a4du);
    // RTXDI walks a precomputed disk-offset sequence from a random start index.
    const int start_index = static_cast<int>(rng_float(restir_rng) * kRestirNeighborOffsetCount) &
        (kRestirNeighborOffsetCount - 1);
    restir_stream_reservoir_gpu(reservoir, canonical, canonical.selected_target,
        kRestirMaxM, canonical.M, restir_rng);
    const int spatial_sample_count = canonical.M < kRestirSpatialTargetHistory
        ? kRestirSpatialDisocclusionSamples : kRestirSpatialSamples;
    GpuRestirSpatialState state;
    state.start_index = start_index;
    state.sample_count = spatial_sample_count;
    int selected_neighbor = -1;
    for (int i = 0; i < spatial_sample_count; ++i) {
        const int offset_index = (start_index + i) & (kRestirNeighborOffsetCount - 1);
        int ox = 0;
        int oy = 0;
        restir_neighbor_offset_gpu(neighbor_offsets, offset_index, ox, oy);
        const int nx = x + ox;
        const int ny = y + oy;
        if (nx < 0 || nx >= settings.width || ny < 0 || ny >= settings.height) continue;
        const int neighbor_pixel = ny * settings.width + nx;
        if (!restir_surfaces_compatible_gpu(current, current_surfaces[neighbor_pixel])) continue;
        const GpuRestirReservoir neighbor = temporal[neighbor_pixel];
        const int available_m = kRestirMaxM - reservoir.M;
        const int neighbor_m = neighbor.M < available_m ? neighbor.M : available_m;
        state.carried_m[i] = neighbor_m;
        Vec3 direction;
        float distance = 0.0f;
        const Vec3 contribution = restir_evaluate_surface_sample_gpu(scene, current, settings,
            neighbor.sample, direction, distance);
        if (restir_stream_reservoir_gpu(reservoir, neighbor, restir_luminance_gpu(contribution),
                kRestirMaxM, neighbor_m, restir_rng)) {
            selected_neighbor = i;
        }
    }
    state.selected_neighbor = selected_neighbor;
    spatial_states[path.pixel] = state;
    output[path.pixel] = reservoir;
}

__global__ void restir_spatial_finalize_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, const GpuRestirReservoir* temporal,
    const GpuRestirSurface* current_surfaces, const Vec2* neighbor_offsets,
    const GpuRestirSpatialState* spatial_states,
    GpuRestirReservoir* output) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const GpuScene& scene = *scene_ptr;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;
    const int pixel = path.pixel;
    const GpuRestirReservoir canonical = temporal[pixel];
    GpuRestirReservoir reservoir = output[pixel];
    const GpuRestirSurface current = current_surfaces[pixel];
    const int x = pixel % settings.width;
    const int y = pixel / settings.width;
    uint32_t restir_rng = restir_rng_seed_gpu(path.rng, 0xb5297a4du ^ 0x9e3779b9u);
    const GpuRestirSpatialState state = spatial_states[pixel];
    if (reservoir.valid) {
        const float current_target = reservoir.selected_target;
        float normalization_numerator = state.selected_neighbor < 0 ? current_target : 0.0f;
        float normalization_denominator = current_target * static_cast<float>(canonical.M);
        for (int i = 0; i < state.sample_count; ++i) {
            const int offset_index = (state.start_index + i) & (kRestirNeighborOffsetCount - 1);
            int ox = 0;
            int oy = 0;
            restir_neighbor_offset_gpu(neighbor_offsets, offset_index, ox, oy);
            const int nx = x + ox;
            const int ny = y + oy;
            if (nx < 0 || nx >= settings.width || ny < 0 || ny >= settings.height) continue;
            const int neighbor_pixel = ny * settings.width + nx;
            const GpuRestirSurface neighbor_surface = current_surfaces[neighbor_pixel];
            if (!restir_surfaces_compatible_gpu(current, neighbor_surface)) continue;
            const int neighbor_m = state.carried_m[i];
            if (neighbor_m <= 0) continue;
            const float neighbor_target = restir_source_target_gpu(scene, neighbor_surface, settings,
                reservoir.sample, settings.cuda_restir_bias_correction == RestirBiasCorrection::RayTraced, restir_rng);
            if (state.selected_neighbor == i) normalization_numerator = neighbor_target;
            normalization_denominator += neighbor_target * static_cast<float>(neighbor_m);
        }
        restir_finalize_gpu(reservoir, normalization_numerator, normalization_denominator);
    }
    output[pixel] = reservoir;
}

__device__ Vec3 restir_evaluate_reservoir_gpu(const GpuScene& scene, const GpuHit& hit, const GpuMaterial& material,
    Vec3 wo, uint32_t& rng, const RenderSettings& settings, GpuRestirReservoir& reservoir, int pixel) {
    if (!reservoir.valid || reservoir.M <= 0 || reservoir.selected_target <= 0.0f || reservoir.weight_sum <= 0.0f) return {};
    Vec3 direction;
    float distance = 0.0f;
    const Vec3 contribution = restir_evaluate_sample_gpu(scene, hit, material, wo, settings, reservoir.sample, direction, distance);
    const float target = restir_luminance_gpu(contribution);
    if (!(target > 0.0f)) return {};
    bool visible = false;
    if (settings.cuda_restir_final_visibility_reuse && reservoir.visibility >= 0 &&
        reservoir.visibility_age <= kRestirFinalVisibilityMaxAge && reservoir.visibility_pixel >= 0) {
        const int x = pixel % settings.width;
        const int y = pixel / settings.width;
        const int previous_x = reservoir.visibility_pixel % settings.width;
        const int previous_y = reservoir.visibility_pixel / settings.width;
        visible = abs(x - previous_x) <= kRestirFinalVisibilityMaxDistance &&
            abs(y - previous_y) <= kRestirFinalVisibilityMaxDistance && reservoir.visibility != 0;
        if (abs(x - previous_x) <= kRestirFinalVisibilityMaxDistance &&
            abs(y - previous_y) <= kRestirFinalVisibilityMaxDistance) {
            if (!visible) {
                // Match RTXDI_StoreVisibilityInDIReservoir(..., discardIfInvisible=true):
                // preserve M but prevent a known-zero sample from entering future reuse.
                reservoir.valid = 0;
                reservoir.weight_sum = 0.0f;
            }
            return visible ? mul(contribution, reservoir.weight_sum) : Vec3{};
        }
    }
    visible = restir_sample_visible_gpu(scene, hit, material, reservoir.sample, direction, distance, rng);
    // Final visibility is useful history even when visibility reuse is disabled.
    // In particular, an occluded sample must not survive temporal/spatial reuse
    // indefinitely; keep M for normalization and discard only the selected sample.
    reservoir.visibility = visible ? 1 : 0;
    reservoir.visibility_age = 0;
    reservoir.visibility_pixel = pixel;
    if (!visible) {
        reservoir.valid = 0;
        reservoir.weight_sum = 0.0f;
        return {};
    }
    return mul(contribution, reservoir.weight_sum);
}

__global__ void restir_generate_visibility_rays_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuHit* hits, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, GpuRestirReservoir* reservoirs,
    GpuRestirVisibilityRay* visibility_rays, int* visibility_results,
    int* visibility_indices, int* visibility_count) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;

    const int pixel = path.pixel;
    visibility_results[pixel] = -1;
    visibility_rays[pixel] = {};
    GpuRestirReservoir& reservoir = reservoirs[pixel];
    if (!reservoir.valid || reservoir.M <= 0 || reservoir.selected_target <= 0.0f ||
        reservoir.weight_sum <= 0.0f) {
        return;
    }

    const GpuScene& scene = *scene_ptr;
    const GpuHit& hit = hits[path_index];
    const GpuMaterial& material = scene.materials[hit.material];
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    Vec3 direction;
    float distance = 0.0f;
    const Vec3 contribution =
        restir_evaluate_sample_gpu(scene, hit, material, wo, settings, reservoir.sample, direction, distance);
    const float target = restir_luminance_gpu(contribution);
    if (!(target > 0.0f)) return;

    if (restir_cached_visibility_valid_gpu(reservoir, settings, pixel)) {
        const bool visible = reservoir.visibility != 0;
        visibility_results[pixel] = visible ? 1 : 0;
        if (!visible) {
            reservoir.valid = 0;
            reservoir.weight_sum = 0.0f;
        }
        return;
    }

    float ndotl_raw = 0.0f;
    restir_surface_ndotl_gpu(material, hit, direction, ndotl_raw);
    GpuRestirVisibilityRay ray;
    ray.ray = {add(hit.position, mul(hit.normal, 0.002f * (ndotl_raw >= 0.0f ? 1.0f : -1.0f))), direction};
    ray.remaining = isfinite(distance) ? distance - 0.01f : kInfinity;
    ray.sample_type = static_cast<int>(reservoir.sample.type);
    ray.sample_index = reservoir.sample.index;
    ray.path_index = path_index;
    ray.valid = 1;
    visibility_rays[pixel] = ray;
    visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
}

template <bool TransparentVisibility, bool TwoLevel, GpuTraversalLayout Layout>
__global__ void restir_trace_visibility_kernel(const GpuScene* scene_ptr, GpuWavefrontPaths paths,
    const int* visibility_indices, const int* visibility_count,
    const GpuRestirVisibilityRay* visibility_rays, int* visibility_results) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= *visibility_count) return;
    const int pixel = visibility_indices[queue_index];
    const GpuRestirVisibilityRay visibility = visibility_rays[pixel];

    const GpuScene& scene = *scene_ptr;
    Ray ray = visibility.ray;
    float remaining = visibility.remaining;
    bool visible = true;
    if constexpr (!TransparentVisibility) {
        GpuCompactHit hit{};
        hit.t = remaining;
        if (intersect_compact_gpu<TwoLevel, Layout>(scene, ray, hit) &&
            !restir_visibility_hit_is_target_gpu(visibility, hit)) {
            visible = false;
        }
        visibility_results[pixel] = visible ? 1 : 0;
    } else {
        GpuWavefrontPathRef path = wavefront_path_ref(paths, visibility.path_index);
        for (int step = 0; step < 8; ++step) {
            GpuCompactHit hit{};
            hit.t = remaining;
            if (!intersect_compact_gpu<TwoLevel, Layout>(scene, ray, hit)) break;
            if (restir_visibility_hit_is_target_gpu(visibility, hit)) {
                break;
            }
            const int material_index = hit.triangle >= 0 ? traversal_material_index(hit.material) : hit.material;
            if (material_index < 0 || material_index >= scene.material_count) {
                visible = false;
                break;
            }
            const GpuMaterial material = scene.materials[material_index];
            const Vec2 uv = compact_hit_uv_gpu(scene, ray, hit);
            if (!material_visible_gpu(scene, material, uv, path.rng) ||
                material.brdf_model == static_cast<int>(BrdfModel::Dielectric) ||
                material_transmission_gpu(scene, material, uv) > 0.5f) {
                const Vec3 position = compact_hit_position_gpu(ray, hit);
                ray = {add(position, mul(ray.direction, 0.002f)), ray.direction};
                if (isfinite(remaining)) {
                    remaining -= hit.t + 0.002f;
                    if (remaining <= 0.001f) break;
                }
                continue;
            }
            visible = false;
            break;
        }
        visibility_results[pixel] = visible ? 1 : 0;
    }
}

__global__ void restir_resolve_visibility_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuHit* hits, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, GpuRestirReservoir* reservoirs,
    const int* visibility_results) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;

    const int pixel = path.pixel;
    GpuRestirReservoir& reservoir = reservoirs[pixel];
    if (!reservoir.valid || reservoir.M <= 0 || reservoir.selected_target <= 0.0f ||
        reservoir.weight_sum <= 0.0f) {
        return;
    }

    const GpuScene& scene = *scene_ptr;
    const GpuHit& hit = hits[path_index];
    const GpuMaterial& material = scene.materials[hit.material];
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    Vec3 direction;
    float distance = 0.0f;
    const Vec3 contribution =
        restir_evaluate_sample_gpu(scene, hit, material, wo, settings, reservoir.sample, direction, distance);
    if (!(restir_luminance_gpu(contribution) > 0.0f)) return;

    const bool visible = visibility_results[pixel] > 0;
    reservoir.visibility = visible ? 1 : 0;
    reservoir.visibility_age = 0;
    reservoir.visibility_pixel = pixel;
    if (!visible) {
        reservoir.valid = 0;
        reservoir.weight_sum = 0.0f;
        return;
    }

    path.radiance = add(path.radiance,
        clamp_sample_radiance_gpu(mul(path.throughput, mul(contribution, reservoir.weight_sum)), 64.0f));
    wavefront_set_restir_direct(path, true);
}

__global__ void restir_resolve_initial_visibility_kernel(GpuWavefrontPaths paths, const int* shadow_indices,
    const GpuWavefrontQueueCounters* queue_counters, GpuRestirReservoir* reservoirs,
    const int* visibility_results) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int shadow_count = queue_counters->num_queued[GpuWavefrontQueueShadow];
    if (queue_index >= shadow_count) return;
    const int path_index = shadow_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;

    const int pixel = path.pixel;
    if (visibility_results[pixel] != 0) return;
    GpuRestirReservoir& reservoir = reservoirs[pixel];
    reservoir.visibility = 0;
    reservoir.visibility_age = 0;
    reservoir.visibility_pixel = pixel;
    reservoir.valid = 0;
    reservoir.weight_sum = 0.0f;
}

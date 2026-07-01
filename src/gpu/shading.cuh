__host__ __device__ bool has_light_emission_gpu(Vec3 emission) {
    return emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
}

__host__ __device__ bool finite_vec_gpu(Vec3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__host__ __device__ Vec3 clamp_sample_radiance_gpu(Vec3 v, float limit = 64.0f) {
    if (!finite_vec_gpu(v)) {
        return {};
    }
    const float max_sample_radiance = fmaxf(0.0f, limit);
    return {
        dclamp(v.x, 0.0f, max_sample_radiance),
        dclamp(v.y, 0.0f, max_sample_radiance),
        dclamp(v.z, 0.0f, max_sample_radiance),
    };
}

__host__ __device__ Vec3 emitted_radiance_gpu(const GpuTriangle& light, const GpuMaterial& material, Vec3 mesh_emission, Vec3 material_emission, bool mesh_light_double_sided, Vec3 ray_direction_to_light) {
    const bool front_facing = ddot(light.normal, mul(ray_direction_to_light, -1.0f)) > 0.0f;
    Vec3 emission{};
    if (has_light_emission_gpu(mesh_emission) && (mesh_light_double_sided || front_facing)) {
        emission = add(emission, mesh_emission);
    }
    if (has_light_emission_gpu(material_emission) && (material.double_sided || front_facing)) {
        emission = add(emission, material_emission);
    }
    return emission;
}

__device__ Vec3 fresnel_schlick_gpu(float cos_theta, Vec3 f0) {
    const float f = powf(dclamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
    return add(f0, mul(sub(Vec3{1.0f, 1.0f, 1.0f}, f0), f));
}

__device__ float fresnel_conductor_channel_gpu(float cos_theta, float eta, float k) {
    const float c = dclamp(fabsf(cos_theta), 0.0f, 1.0f);
    const float cos2 = c * c;
    const float sin2 = fmaxf(0.0f, 1.0f - cos2);
    eta = fmaxf(0.0f, eta);
    k = fmaxf(0.0f, k);
    const float eta2 = eta * eta;
    const float k2 = k * k;
    const float t0 = eta2 - k2 - sin2;
    const float a2pb2 = sqrtf(fmaxf(0.0f, t0 * t0 + 4.0f * eta2 * k2));
    const float a = sqrtf(fmaxf(0.0f, 0.5f * (a2pb2 + t0)));
    const float t1 = a2pb2 + cos2;
    const float t2 = 2.0f * c * a;
    const float rs = (t1 - t2) / fmaxf(1.0e-6f, t1 + t2);
    const float t3 = cos2 * a2pb2 + sin2 * sin2;
    const float t4 = t2 * sin2;
    const float rp = rs * (t3 - t4) / fmaxf(1.0e-6f, t3 + t4);
    return dclamp(0.5f * (rp + rs), 0.0f, 1.0f);
}

__device__ Vec3 fresnel_conductor_gpu(float cos_theta, Vec3 eta, Vec3 k) {
    return {
        fresnel_conductor_channel_gpu(cos_theta, eta.x, k.x),
        fresnel_conductor_channel_gpu(cos_theta, eta.y, k.y),
        fresnel_conductor_channel_gpu(cos_theta, eta.z, k.z),
    };
}

__device__ Vec3 sample_texture_gpu(const GpuScene& scene, int texture_index, Vec2 uv) {
    if (texture_index < 0 || texture_index >= scene.texture_count) {
        return {1.0f, 1.0f, 1.0f};
    }
    const GpuTexture texture = scene.textures[texture_index];
    if (texture.width <= 0 || texture.height <= 0) {
        return {1.0f, 1.0f, 1.0f};
    }
    if (texture.object == 0) {
        return {1.0f, 1.0f, 1.0f};
    }
    const float4 sample = tex2D<float4>(texture.object, wrap01_gpu(uv.x), wrap01_gpu(1.0f - uv.y));
    return {sample.x, sample.y, sample.z};
}

__device__ Vec3 sample_texture_lod_gpu(const GpuScene& scene, int texture_index, Vec2 uv, float lod) {
    if (texture_index < 0 || texture_index >= scene.texture_count) {
        return {1.0f, 1.0f, 1.0f};
    }
    const GpuTexture texture = scene.textures[texture_index];
    if (texture.width <= 0 || texture.height <= 0 || texture.object == 0) {
        return {1.0f, 1.0f, 1.0f};
    }
    const int last_level = texture.mip_levels > 1 ? texture.mip_levels - 1 : 0;
    lod = dclamp(lod, 0.0f, static_cast<float>(last_level));
    const float4 sample = tex2DLod<float4>(texture.object, wrap01_gpu(uv.x), wrap01_gpu(1.0f - uv.y), lod);
    return {sample.x, sample.y, sample.z};
}

__device__ Vec2 wrap_oct_sphere_gpu(Vec2 uv) {
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
    return {dclamp(u, 0.0f, 1.0f), dclamp(v, 0.0f, 1.0f)};
}

__device__ Vec3 sample_environment_texture_lod_gpu(const GpuScene& scene, int texture_index, Vec2 uv, float lod, int mapping) {
    if (texture_index < 0 || texture_index >= scene.texture_count) {
        return {1.0f, 1.0f, 1.0f};
    }
    const GpuTexture texture = scene.textures[texture_index];
    if (texture.width <= 0 || texture.height <= 0 || texture.object == 0) {
        return {1.0f, 1.0f, 1.0f};
    }
    const int last_level = texture.mip_levels > 1 ? texture.mip_levels - 1 : 0;
    lod = dclamp(lod, 0.0f, static_cast<float>(last_level));
    uv = mapping == 1 ? wrap_oct_sphere_gpu(uv) : Vec2{wrap01_gpu(uv.x), dclamp(uv.y, 0.0f, 1.0f)};
    const float image_v = mapping == 1 ? uv.y : 1.0f - uv.y;
    const float4 sample = tex2DLod<float4>(texture.object, uv.x, image_v, lod);
    return {sample.x, sample.y, sample.z};
}

__device__ float sample_texture_alpha_gpu(const GpuScene& scene, int texture_index, Vec2 uv) {
    if (texture_index < 0 || texture_index >= scene.texture_count) {
        return 1.0f;
    }
    const GpuTexture texture = scene.textures[texture_index];
    if (texture.width <= 0 || texture.height <= 0 || texture.object == 0) {
        return 1.0f;
    }
    const float4 sample = tex2D<float4>(texture.object, wrap01_gpu(uv.x), wrap01_gpu(1.0f - uv.y));
    return sample.w;
}

__device__ Vec2 transform_uv_gpu(Vec2 uv, Vec2 offset, Vec2 scale, float rotation) {
    uv.x *= scale.x;
    uv.y *= scale.y;
    if (rotation != 0.0f) {
        const float c = cosf(rotation);
        const float s = sinf(rotation);
        uv = {uv.x * c - uv.y * s, uv.x * s + uv.y * c};
    }
    uv.x += offset.x;
    uv.y += offset.y;
    return uv;
}

__device__ Vec2 material_base_uv_gpu(const GpuMaterial& material, Vec2 uv) {
    return transform_uv_gpu(uv, material.texture_offset, material.texture_scale, material.texture_rotation);
}

__device__ Vec2 material_emission_uv_gpu(const GpuMaterial& material, Vec2 uv) {
    return transform_uv_gpu(uv, material.emission_texture_offset, material.emission_texture_scale, material.emission_texture_rotation);
}

__device__ Vec3 material_emission_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return mul(material.emission, sample_texture_gpu(scene, material.emission_texture_index, material_emission_uv_gpu(material, uv)));
}

__device__ float material_opacity_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return dclamp(material.alpha * sample_texture_alpha_gpu(scene, material.texture_index, material_base_uv_gpu(material, uv)), 0.0f, 1.0f);
}

__device__ bool material_visible_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv, uint32_t& rng) {
    if (material.alpha_mode == static_cast<int>(AlphaMode::Opaque)) {
        return true;
    }
    const float opacity = material_opacity_gpu(scene, material, uv);
    if (material.alpha_mode == static_cast<int>(AlphaMode::Mask)) {
        return opacity >= material.alpha_cutoff;
    }
    return rng_float(rng) <= opacity;
}

__device__ Vec3 apply_normal_map_gpu(const GpuScene& scene, const GpuMaterial& material, const GpuHit& hit, Vec3 ray_direction) {
    if (material.normal_texture_index < 0) {
        return hit.normal;
    }
    const Vec3 s = sub(mul(sample_texture_gpu(scene, material.normal_texture_index, hit.uv), 2.0f), Vec3{1.0f, 1.0f, 1.0f});
    Vec3 tangent = dnormalize(sub(hit.tangent, mul(hit.normal, ddot(hit.normal, hit.tangent))));
    if (ddot(tangent, tangent) <= 0.0f) {
        tangent = dnormalize(dcross(fabsf(hit.normal.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f}, hit.normal));
    }
    const Vec3 bitangent = ddot(hit.bitangent, hit.bitangent) > 0.0f ? hit.bitangent : dcross(hit.normal, tangent);
    Vec3 mapped = dnormalize(add(add(mul(tangent, s.x * material.normal_scale), mul(bitangent, s.y * material.normal_scale)), mul(hit.normal, s.z)));
    if (ddot(mapped, mapped) <= 0.0f) {
        mapped = hit.normal;
    }
    return ddot(mapped, ray_direction) < 0.0f ? mapped : mul(mapped, -1.0f);
}

__device__ float material_roughness_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    if (material.brdf_model == static_cast<int>(BrdfModel::Conductor)) {
        return dclamp(material.roughness, 0.0f, 1.0f);
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::StandardSurface) && material.roughness_texture_index >= 0) {
        return dclamp(material.roughness * sample_texture_gpu(scene, material.roughness_texture_index, uv).x, 0.02f, 1.0f);
    }
    const Vec3 metallic_roughness = sample_texture_gpu(scene, material.metallic_roughness_texture_index, uv);
    return dclamp(material.roughness * metallic_roughness.y, 0.02f, 1.0f);
}

__device__ float material_metallic_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    if (material.brdf_model == static_cast<int>(BrdfModel::StandardSurface) && material.metallic_texture_index >= 0) {
        return dclamp(material.metallic * sample_texture_gpu(scene, material.metallic_texture_index, uv).x, 0.0f, 1.0f);
    }
    const Vec3 metallic_roughness = sample_texture_gpu(scene, material.metallic_roughness_texture_index, uv);
    return dclamp(material.metallic * metallic_roughness.z, 0.0f, 1.0f);
}

__device__ float material_specular_weight_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return fmaxf(0.0f, material.specular_weight * sample_texture_gpu(scene, material.specular_texture_index, uv).x);
}

__device__ Vec3 material_sheen_color_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return mul(material.sheen_color, sample_texture_gpu(scene, material.sheen_color_texture_index, uv));
}

__device__ float material_clearcoat_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return dclamp(material.clearcoat * sample_texture_gpu(scene, material.clearcoat_texture_index, uv).x, 0.0f, 1.0f);
}

__device__ float material_clearcoat_roughness_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return dclamp(material.clearcoat_roughness * sample_texture_gpu(scene, material.clearcoat_roughness_texture_index, uv).y, 0.02f, 1.0f);
}

__device__ float material_transmission_gpu(const GpuScene& scene, const GpuMaterial& material, Vec2 uv) {
    return dclamp(material.transmission * sample_texture_gpu(scene, material.transmission_texture_index, uv).x, 0.0f, 1.0f);
}

__device__ Vec2 equal_area_sphere_to_square_gpu(Vec3 direction) {
    const float ax = fabsf(direction.x);
    const float ay = fabsf(direction.y);
    const float az = fabsf(direction.z);
    const float r = sqrtf(fmaxf(0.0f, 1.0f - az));
    if (r <= 1.0e-8f) {
        return {0.5f, direction.z >= 0.0f ? 0.5f : 0.0f};
    }
    const float phi = atan2f(ay, ax);
    const float a = 4.0f * phi / kPi - 1.0f;
    const float sum = direction.z >= 0.0f ? r : 2.0f - r;
    const float diff = a * r;
    const float up = (sum - diff) * 0.5f;
    const float vp = (sum + diff) * 0.5f;
    const float u = direction.x < 0.0f ? -up : up;
    const float v = direction.y < 0.0f ? -vp : vp;
    return {u * 0.5f + 0.5f, v * 0.5f + 0.5f};
}

__device__ float environment_mip_lod_gpu(const GpuScene& scene, const GpuTexture& texture, const RenderSettings& settings) {
    if (texture.width <= 0 || texture.height <= 0 || settings.width <= 0 || settings.height <= 0) {
        return 0.0f;
    }
    const float aspect = static_cast<float>(settings.width) / static_cast<float>(settings.height);
    const float half_height = tanf(scene.camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const float ray_solid_angle = fmaxf(1.0e-8f, (4.0f * half_width * half_height) / static_cast<float>(settings.width * settings.height));
    const float texel_solid_angle = 4.0f * kPi / static_cast<float>(texture.width * texture.height);
    return fmaxf(0.0f, 0.5f * log2f(ray_solid_angle / fmaxf(1.0e-12f, texel_solid_angle)));
}

__device__ Vec3 environment_radiance_gpu(const GpuScene& scene, Vec3 direction, const RenderSettings& settings) {
    if (scene.environment_texture >= 0) {
        direction = dnormalize({
            ddot(direction, scene.environment_light_from_world_x),
            ddot(direction, scene.environment_light_from_world_y),
            ddot(direction, scene.environment_light_from_world_z),
        });
        Vec2 uv;
        if (scene.environment_mapping == 1) {
            uv = equal_area_sphere_to_square_gpu(direction);
        } else {
            const float u = atan2f(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
            const float v = acosf(dclamp(direction.y, -1.0f, 1.0f)) / kPi;
            uv = {u, 1.0f - v};
        }
        const GpuTexture texture = scene.textures[scene.environment_texture];
        return mul(mul(scene.environment_color, sample_environment_texture_lod_gpu(scene, scene.environment_texture, uv, environment_mip_lod_gpu(scene, texture, settings), scene.environment_mapping)), scene.environment_strength);
    }
    if (scene.environment_constant) {
        return mul(scene.environment_color, scene.environment_strength);
    }
    const float t = 0.5f * (direction.y + 1.0f);
    return mul(mul(add(mul(Vec3{0.02f, 0.025f, 0.035f}, 1.0f - t), mul(Vec3{0.32f, 0.45f, 0.68f}, t)), scene.environment_color), scene.environment_strength);
}

__device__ float ggx_distribution_gpu(float ndoth, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(1.0e-6f, kPi * d * d);
}

__device__ float ggx_distribution_alpha_gpu(float ndoth, float alpha) {
    const float a2 = alpha * alpha;
    const float d = ndoth * ndoth * (a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(1.0e-6f, kPi * d * d);
}

__device__ float smith_ggx_g1_gpu(float ndotv, float roughness) {
    const float r = roughness + 1.0f;
    const float k = (r * r) * 0.125f;
    return ndotv / fmaxf(1.0e-6f, ndotv * (1.0f - k) + k);
}

__device__ float smith_ggx_g1_alpha_gpu(float ndotv, float alpha) {
    if (ndotv <= 0.0f) {
        return 0.0f;
    }
    const float cos2 = ndotv * ndotv;
    const float tan2 = fmaxf(0.0f, 1.0f - cos2) / fmaxf(1.0e-6f, cos2);
    return 2.0f / (1.0f + sqrtf(1.0f + alpha * alpha * tan2));
}

__device__ float charlie_sheen_distribution_gpu(float ndoth, float roughness) {
    const float alpha = dclamp(roughness, 0.01f, 1.0f);
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - ndoth * ndoth));
    return (2.0f + 1.0f / alpha) * powf(sin_theta, 1.0f / alpha) / (2.0f * kPi);
}

__device__ float sheen_visibility_gpu(float ndotv, float ndotl) {
    return 1.0f / fmaxf(1.0e-6f, 4.0f * (ndotl + ndotv - ndotl * ndotv));
}

__device__ Vec3 evaluate_brdf_gpu(const GpuScene& scene, const GpuMaterial& material, Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) {
    const float ndotv = material.double_sided ? fabsf(ddot(n, wo)) : fmaxf(0.0f, ddot(n, wo));
    const float ndotl = material.double_sided ? fabsf(ddot(n, wi)) : fmaxf(0.0f, ddot(n, wi));
    if (ndotv <= 0.0f || ndotl <= 0.0f) {
        return {};
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Lambertian)) {
        return divv(mul(material.albedo, sample_texture_gpu(scene, material.texture_index, material_base_uv_gpu(material, uv))), kPi);
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Conductor)) {
        const float alpha = material_roughness_gpu(scene, material, uv);
        if (alpha <= 0.001f) {
            return {};
        }
        const Vec3 facing_n = ddot(n, wo) >= 0.0f ? n : mul(n, -1.0f);
        const Vec3 h = dnormalize(add(wi, wo));
        const float ndoth = fmaxf(0.0f, ddot(facing_n, h));
        const float vdoth = fmaxf(0.0f, ddot(wo, h));
        if (ndoth <= 0.0f || vdoth <= 0.0f) {
            return {};
        }
        const Vec3 color = mul(material.albedo, sample_texture_gpu(scene, material.texture_index, material_base_uv_gpu(material, uv)));
        const Vec3 f = mul(color, fresnel_conductor_gpu(vdoth, material.conductor_eta, material.conductor_k));
        const float d = ggx_distribution_alpha_gpu(ndoth, dclamp(alpha, 0.001f, 1.0f));
        const float g = smith_ggx_g1_alpha_gpu(ndotv, alpha) * smith_ggx_g1_alpha_gpu(ndotl, alpha);
        return mul(f, d * g / fmaxf(1.0e-6f, 4.0f * ndotv * ndotl));
    }
    const bool standard_surface = material.brdf_model == static_cast<int>(BrdfModel::StandardSurface);
    if (material.brdf_model != static_cast<int>(BrdfModel::Principled) && !standard_surface) {
        return {};
    }

    const float roughness = material_roughness_gpu(scene, material, uv);
    const float metallic = material_metallic_gpu(scene, material, uv);
    const float transmission = standard_surface ? material_transmission_gpu(scene, material, uv) : 0.0f;
    const float sheen_roughness = dclamp(material.sheen_roughness * sample_texture_alpha_gpu(scene, material.sheen_roughness_texture_index, uv), 0.01f, 1.0f);
    const float clearcoat = material_clearcoat_gpu(scene, material, uv);
    const float clearcoat_roughness = material_clearcoat_roughness_gpu(scene, material, uv);
    const Vec3 h = dnormalize(add(wi, wo));
    const float ndoth = fmaxf(0.0f, ddot(n, h));
    const float vdoth = fmaxf(0.0f, ddot(wo, h));
    const Vec3 color = mul(material.albedo, sample_texture_gpu(scene, material.texture_index, material_base_uv_gpu(material, uv)));
    const float specular_weight = standard_surface ? material_specular_weight_gpu(scene, material, uv) : 1.0f;
    const float dielectric_f0 = standard_surface
        ? powf((1.0f - material.specular_ior) / (1.0f + material.specular_ior), 2.0f) * specular_weight
        : 0.04f;
    const Vec3 f0 = dlerp(Vec3{dielectric_f0, dielectric_f0, dielectric_f0}, color, metallic);
    const Vec3 f = fresnel_schlick_gpu(vdoth, f0);
    const float d = ggx_distribution_gpu(ndoth, roughness);
    const float g = smith_ggx_g1_gpu(ndotv, roughness) * smith_ggx_g1_gpu(ndotl, roughness);
    const Vec3 specular = mul(f, d * g / fmaxf(1.0e-6f, 4.0f * ndotv * ndotl));
    const Vec3 diffuse = divv(mul(color, (1.0f - metallic) * (1.0f - transmission)), kPi);
    const Vec3 sheen = mul(material_sheen_color_gpu(scene, material, uv), charlie_sheen_distribution_gpu(ndoth, sheen_roughness) * sheen_visibility_gpu(ndotv, ndotl));
    const Vec3 coat_f = fresnel_schlick_gpu(vdoth, Vec3{0.04f, 0.04f, 0.04f});
    const float coat_d = ggx_distribution_gpu(ndoth, clearcoat_roughness);
    const float coat_g = smith_ggx_g1_gpu(ndotv, clearcoat_roughness) * smith_ggx_g1_gpu(ndotl, clearcoat_roughness);
    const Vec3 clearcoat_lobe = mul(coat_f, clearcoat * coat_d * coat_g / fmaxf(1.0e-6f, 4.0f * ndotv * ndotl));
    return add(add(add(diffuse, specular), sheen), clearcoat_lobe);
}

__device__ Vec3 reflect_gpu(Vec3 v, Vec3 n) {
    return sub(mul(n, 2.0f * ddot(v, n)), v);
}

__device__ Vec3 refract_gpu(Vec3 v, Vec3 n, float eta) {
    const float cos_theta = fminf(ddot(v, n), 1.0f);
    const Vec3 perpendicular = mul(sub(mul(n, cos_theta), v), eta);
    const float parallel_len2 = fmaxf(0.0f, 1.0f - ddot(perpendicular, perpendicular));
    return dnormalize(sub(perpendicular, mul(n, sqrtf(parallel_len2))));
}

__device__ float fresnel_dielectric_gpu(float cos_theta, float eta) {
    float r0 = (1.0f - eta) / (1.0f + eta);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(dclamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
}

__device__ Vec3 sample_ggx_half_gpu(Vec3 n, float roughness, float u1, float u2) {
    const float a = roughness * roughness;
    const float phi = 2.0f * kPi * u1;
    const float cos_theta = sqrtf((1.0f - u2) / fmaxf(1.0e-6f, 1.0f + (a * a - 1.0f) * u2));
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    return dnormalize(to_world_gpu({sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta}, n));
}

__device__ Vec3 sample_ggx_half_alpha_gpu(Vec3 n, float alpha, float u1, float u2) {
    const float a2 = alpha * alpha;
    const float phi = 2.0f * kPi * u1;
    const float cos_theta = sqrtf((1.0f - u2) / fmaxf(1.0e-6f, 1.0f + (a2 - 1.0f) * u2));
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    return dnormalize(to_world_gpu({sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta}, n));
}

__device__ float material_pdf_gpu(const GpuScene& scene, const GpuMaterial& material, Vec3 n, Vec3 wo, Vec3 wi, Vec2 uv) {
    const float ndotl = material.double_sided ? fabsf(ddot(n, wi)) : fmaxf(0.0f, ddot(n, wi));
    const float ndotv = material.double_sided ? fabsf(ddot(n, wo)) : fmaxf(0.0f, ddot(n, wo));
    if (ndotl <= 0.0f || ndotv <= 0.0f) {
        return 0.0f;
    }
    const float diffuse_pdf = ndotl / kPi;
    if (material.brdf_model == static_cast<int>(BrdfModel::Lambertian)) {
        return diffuse_pdf;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Conductor)) {
        const float alpha = material_roughness_gpu(scene, material, uv);
        if (alpha <= 0.001f) {
            return 0.0f;
        }
        const Vec3 facing_n = ddot(n, wo) >= 0.0f ? n : mul(n, -1.0f);
        const Vec3 h = dnormalize(add(wi, wo));
        const float ndoth = fmaxf(0.0f, ddot(facing_n, h));
        const float vdoth = fmaxf(0.0f, ddot(wo, h));
        if (ndoth <= 0.0f || vdoth <= 0.0f) {
            return 0.0f;
        }
        return ggx_distribution_alpha_gpu(ndoth, dclamp(alpha, 0.001f, 1.0f)) * ndoth / fmaxf(1.0e-6f, 4.0f * vdoth);
    }
    const bool standard_surface = material.brdf_model == static_cast<int>(BrdfModel::StandardSurface);
    if (material.brdf_model != static_cast<int>(BrdfModel::Principled) && !standard_surface) {
        return 0.0f;
    }
    const float roughness = material_roughness_gpu(scene, material, uv);
    const float metallic = material_metallic_gpu(scene, material, uv);
    const float transmission = standard_surface ? material_transmission_gpu(scene, material, uv) : 0.0f;
    const float specular_weight = standard_surface ? material_specular_weight_gpu(scene, material, uv) : 1.0f;
    const float spec_prob = dclamp(0.05f + 0.20f * specular_weight + 0.75f * metallic + 0.25f * material_clearcoat_gpu(scene, material, uv), 0.05f, 1.0f);
    const Vec3 h = dnormalize(add(wi, wo));
    const float ndoth = fmaxf(0.0f, ddot(n, h));
    const float vdoth = fmaxf(0.0f, ddot(wo, h));
    const float specular_pdf = ggx_distribution_gpu(ndoth, roughness) * ndoth / fmaxf(1.0e-6f, 4.0f * vdoth);
    return transmission + (1.0f - transmission) * ((1.0f - spec_prob) * diffuse_pdf + spec_prob * specular_pdf);
}

__device__ GpuMaterialSample sample_material_gpu(const GpuScene& scene, const GpuMaterial& material, Vec3 n, Vec3 wo, Vec2 uv, bool front_face, uint32_t& rng) {
    GpuMaterialSample result;
    const Vec3 color = mul(material.albedo, sample_texture_gpu(scene, material.texture_index, material_base_uv_gpu(material, uv)));
    if (material.brdf_model == static_cast<int>(BrdfModel::Mirror)) {
        result.direction = dnormalize(reflect_gpu(wo, n));
        result.weight = Vec3{1.0f, 1.0f, 1.0f};
        result.pdf = 1.0f;
        result.delta = true;
        return result;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Conductor)) {
        const Vec3 facing_n = ddot(n, wo) >= 0.0f ? n : mul(n, -1.0f);
        const float alpha = material_roughness_gpu(scene, material, uv);
        if (alpha <= 0.001f) {
            result.direction = dnormalize(reflect_gpu(wo, facing_n));
            result.weight = mul(color, fresnel_conductor_gpu(ddot(facing_n, wo), material.conductor_eta, material.conductor_k));
            result.pdf = 1.0f;
            result.delta = true;
            return result;
        }
        const Vec3 h = sample_ggx_half_alpha_gpu(facing_n, dclamp(alpha, 0.001f, 1.0f), rng_float(rng), rng_float(rng));
        result.direction = dnormalize(reflect_gpu(wo, h));
        if (ddot(result.direction, facing_n) <= 0.0f) {
            result.pdf = 0.0f;
            result.weight = {};
            return result;
        }
        result.pdf = material_pdf_gpu(scene, material, n, wo, result.direction, uv);
        const float ndotl = fmaxf(0.0f, ddot(facing_n, result.direction));
        result.weight = result.pdf > 0.0f ? mul(evaluate_brdf_gpu(scene, material, n, wo, result.direction, uv), ndotl / result.pdf) : Vec3{};
        return result;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Dielectric)) {
        const float ior = dclamp(material.roughness, 1.0f, 3.0f);
        const float eta = front_face ? 1.0f / ior : ior;
        const float cos_theta = fminf(ddot(wo, n), 1.0f);
        const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
        const bool cannot_refract = eta * sin_theta > 1.0f;
        const float reflectance = fresnel_dielectric_gpu(cos_theta, eta);
        result.direction = (cannot_refract || reflectance > rng_float(rng)) ? dnormalize(reflect_gpu(wo, n)) : refract_gpu(wo, n, eta);
        result.weight = material.transmission_tint;
        result.pdf = 1.0f;
        result.delta = true;
        return result;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::StandardSurface)) {
        const float trans = material_transmission_gpu(scene, material, uv);
        if (trans > 0.0f && rng_float(rng) < trans) {
            const float ior = dclamp(material.specular_ior, 1.0f, 3.0f);
            const float eta = front_face ? 1.0f / ior : ior;
            const float cos_theta = fminf(ddot(wo, n), 1.0f);
            const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
            const bool cannot_refract = eta * sin_theta > 1.0f;
            const float reflectance = fresnel_dielectric_gpu(cos_theta, eta);
            result.direction = (cannot_refract || reflectance > rng_float(rng)) ? dnormalize(reflect_gpu(wo, n)) : refract_gpu(wo, n, eta);
            result.weight = material.transmission_tint;
            result.pdf = fmaxf(1.0e-6f, trans);
            result.delta = true;
            return result;
        }
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::Principled) || material.brdf_model == static_cast<int>(BrdfModel::StandardSurface)) {
        const float roughness = material_roughness_gpu(scene, material, uv);
        const float metallic = material_metallic_gpu(scene, material, uv);
        const bool standard_surface = material.brdf_model == static_cast<int>(BrdfModel::StandardSurface);
        const float specular_weight = standard_surface ? material_specular_weight_gpu(scene, material, uv) : 1.0f;
        const float spec_prob = dclamp(0.05f + 0.20f * specular_weight + 0.75f * metallic + 0.25f * material_clearcoat_gpu(scene, material, uv), 0.05f, 1.0f);
        if (rng_float(rng) < spec_prob) {
            const Vec3 h = sample_ggx_half_gpu(n, roughness, rng_float(rng), rng_float(rng));
            result.direction = dnormalize(reflect_gpu(wo, h));
            if (ddot(result.direction, n) <= 0.0f) {
                result.direction = dnormalize(to_world_gpu(cosine_sample(rng_float(rng), rng_float(rng)), n));
            }
        } else {
            result.direction = dnormalize(to_world_gpu(cosine_sample(rng_float(rng), rng_float(rng)), n));
        }
    } else {
        result.direction = dnormalize(to_world_gpu(cosine_sample(rng_float(rng), rng_float(rng)), n));
    }
    result.pdf = material_pdf_gpu(scene, material, n, wo, result.direction, uv);
    const float ndotl = fmaxf(0.0f, ddot(n, result.direction));
    result.weight = result.pdf > 0.0f ? mul(evaluate_brdf_gpu(scene, material, n, wo, result.direction, uv), ndotl / result.pdf) : Vec3{};
    return result;
}

__device__ Vec3 sample_triangle_area_gpu(const GpuTriangle& tri, uint32_t& rng) {
    const float su = sqrtf(rng_float(rng));
    const float v = rng_float(rng);
    return add(add(mul(tri.v0, 1.0f - su), mul(tri.v1, su * (1.0f - v))), mul(tri.v2, su * v));
}

__device__ Vec2 sample_triangle_uv_gpu(const GpuTriangle& tri, uint32_t& rng, Vec3& point) {
    const float su = sqrtf(rng_float(rng));
    const float v = rng_float(rng);
    const float b0 = 1.0f - su;
    const float b1 = su * (1.0f - v);
    const float b2 = su * v;
    point = add(add(mul(tri.v0, b0), mul(tri.v1, b1)), mul(tri.v2, b2));
    return add(add(mul(tri.uv0, b0), mul(tri.uv1, b1)), mul(tri.uv2, b2));
}

__device__ float triangle_area_gpu(const GpuTriangle& tri) {
    const Vec3 edge1 = sub(tri.v1, tri.v0);
    const Vec3 edge2 = sub(tri.v2, tri.v0);
    return 0.5f * sqrtf(ddot(dcross(edge1, edge2), dcross(edge1, edge2)));
}

__device__ float light_pdf_solid_angle_gpu(const GpuTriangle& light, const GpuMaterial& material, Vec3 origin, Vec3 light_point, float light_pmf) {
    const Vec3 to_light = sub(light_point, origin);
    const float dist2 = ddot(to_light, to_light);
    if (dist2 <= 1.0e-8f) {
        return 0.0f;
    }
    const Vec3 light_dir = divv(to_light, sqrtf(dist2));
    const float ldot_raw = ddot(mul(light.normal, -1.0f), light_dir);
    const bool material_emissive_double_sided = !has_light_emission_gpu(light.emission) && material.double_sided && has_light_emission_gpu(material.emission);
    const float ldot = light.light_double_sided != 0 || material_emissive_double_sided ? fabsf(ldot_raw) : fmaxf(0.0f, ldot_raw);
    const float area = triangle_area_gpu(light);
    return ldot > 0.0f && area > 0.0f && light_pmf > 0.0f ? light_pmf * dist2 / (ldot * area) : 0.0f;
}

__device__ float mis_weight_gpu(float pdf_a, float pdf_b, int heuristic) {
    if (heuristic == static_cast<int>(MisHeuristic::Balance)) {
        return pdf_a / fmaxf(1.0e-8f, pdf_a + pdf_b);
    }
    const float a2 = pdf_a * pdf_a;
    const float b2 = pdf_b * pdf_b;
    return a2 / fmaxf(1.0e-8f, a2 + b2);
}

__device__ Vec3 estimate_direct_gpu(const GpuScene& scene, const GpuHit& hit, const GpuMaterial& material, Vec3 wo, uint32_t& rng, const RenderSettings& settings) {
    Vec3 direct{};
    if (scene.light_count > 0) {
        const int list_index = iclamp_gpu(static_cast<int>(rng_float(rng) * static_cast<float>(scene.light_count)), 0, scene.light_count - 1);
        const int light_index = scene.light_indices[list_index];
        if (light_index >= 0 && light_index < scene.triangle_count) {
            const GpuTriangle light = scene.triangles[light_index];
            Vec3 light_point;
            const Vec2 light_uv = sample_triangle_uv_gpu(light, rng, light_point);
            const Vec3 to_light = sub(light_point, hit.position);
            const float dist2 = ddot(to_light, to_light);
            if (dist2 > 1.0e-8f) {
                const float dist = sqrtf(dist2);
                const Vec3 light_dir = divv(to_light, dist);
                const float ndotl_raw = ddot(hit.normal, light_dir);
                const float ndotl = material.double_sided ? fabsf(ndotl_raw) : fmaxf(0.0f, ndotl_raw);
                const GpuMaterial light_material = scene.materials[light.material];
                const float ldot_raw = ddot(mul(light.normal, -1.0f), light_dir);
                const bool mesh_light_double_sided = light.light_double_sided != 0;
                const bool material_emissive_double_sided = !has_light_emission_gpu(light.emission) && light_material.double_sided && has_light_emission_gpu(material_emission_gpu(scene, light_material, light_uv));
                const float ldot = mesh_light_double_sided || material_emissive_double_sided ? fabsf(ldot_raw) : fmaxf(0.0f, ldot_raw);
                if (ndotl > 0.0f && ldot > 0.0f) {
                    bool blocked = false;
                    const float shadow_offset_side = ndotl_raw >= 0.0f ? 1.0f : -1.0f;
                    Ray shadow_ray{add(hit.position, mul(hit.normal, 0.002f * shadow_offset_side)), light_dir};
                    for (int shadow_step = 0; shadow_step < 8; ++shadow_step) {
                        GpuHit shadow_hit;
                        if (!intersect_gpu(scene, shadow_ray, shadow_hit)) {
                            break;
                        }
                        const float shadow_dist = sqrtf(ddot(sub(shadow_hit.position, hit.position), sub(shadow_hit.position, hit.position)));
                        if (shadow_dist >= dist - 0.01f || shadow_hit.triangle == light_index) {
                            break;
                        }
                        const GpuMaterial shadow_material = scene.materials[shadow_hit.material];
                        if (!material_visible_gpu(scene, shadow_material, shadow_hit.uv, rng) ||
                            shadow_material.brdf_model == static_cast<int>(BrdfModel::Dielectric) ||
                            material_transmission_gpu(scene, shadow_material, shadow_hit.uv) > 0.5f) {
                            shadow_ray = {add(shadow_hit.position, mul(light_dir, 0.002f)), light_dir};
                            continue;
                        }
                        blocked = true;
                        break;
                    }
                    const float light_pmf = 1.0f / static_cast<float>(scene.light_count);
                    const float light_pdf = light_pdf_solid_angle_gpu(light, light_material, hit.position, light_point, light_pmf);
                    const Vec3 light_emission = emitted_radiance_gpu(light, light_material, light.emission, material_emission_gpu(scene, light_material, light_uv), light.light_double_sided != 0, light_dir);
                    if (!blocked && isfinite(light_pdf) && light_pdf > 0.0f && has_light_emission_gpu(light_emission)) {
                        const float bsdf_pdf = material_pdf_gpu(scene, material, hit.normal, wo, light_dir, hit.uv);
                        if (isfinite(bsdf_pdf) && bsdf_pdf >= 0.0f) {
                            const float weight = settings.use_mis ? mis_weight_gpu(light_pdf, bsdf_pdf, static_cast<int>(settings.mis_heuristic)) : 1.0f;
                            direct = add(direct, clamp_sample_radiance_gpu(mul(mul(evaluate_brdf_gpu(scene, material, hit.normal, wo, light_dir, hit.uv), light_emission), ndotl * weight / light_pdf)));
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < scene.directional_light_count; ++i) {
        const GpuDirectionalLight light = scene.directional_lights[i];
        if (light.intensity <= 0.0f || ddot(light.direction, light.direction) <= 0.0f) {
            continue;
        }
        const Vec3 light_dir = dnormalize(light.direction);
        const float ndotl_raw = ddot(hit.normal, light_dir);
        const float ndotl = material.double_sided ? fabsf(ndotl_raw) : fmaxf(0.0f, ndotl_raw);
        if (ndotl <= 0.0f) {
            continue;
        }
        bool blocked = false;
        const float shadow_offset_side = ndotl_raw >= 0.0f ? 1.0f : -1.0f;
        Ray shadow_ray{add(hit.position, mul(hit.normal, 0.002f * shadow_offset_side)), light_dir};
        for (int shadow_step = 0; shadow_step < 8; ++shadow_step) {
            GpuHit shadow_hit;
            if (!intersect_gpu(scene, shadow_ray, shadow_hit)) {
                break;
            }
            const GpuMaterial shadow_material = scene.materials[shadow_hit.material];
            if (!material_visible_gpu(scene, shadow_material, shadow_hit.uv, rng) ||
                shadow_material.brdf_model == static_cast<int>(BrdfModel::Dielectric) ||
                material_transmission_gpu(scene, shadow_material, shadow_hit.uv) > 0.5f) {
                shadow_ray = {add(shadow_hit.position, mul(light_dir, 0.002f)), light_dir};
                continue;
            }
            blocked = true;
            break;
        }
        if (!blocked) {
            direct = add(direct, clamp_sample_radiance_gpu(mul(mul(evaluate_brdf_gpu(scene, material, hit.normal, wo, light_dir, hit.uv), mul(light.color, light.intensity)), ndotl)));
        }
    }
    return direct;
}

__device__ Vec3 estimate_direct_environment_gpu(const GpuScene& scene, const GpuHit& hit, const GpuMaterial& material, Vec3 wo, uint32_t& rng, const RenderSettings& settings) {
    const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal, wo, hit.uv, hit.front_face, rng);
    if (!isfinite(sample.pdf) || sample.pdf <= 0.0f || !finite_vec_gpu(sample.weight) || ddot(sample.weight, sample.weight) <= 0.0f) {
        return {};
    }
    const float offset_side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
    const Ray env_ray{add(hit.position, mul(hit.normal, 0.001f * offset_side)), sample.direction};
    GpuHit env_hit;
    if (intersect_gpu(scene, env_ray, env_hit)) {
        return {};
    }
    return clamp_sample_radiance_gpu(mul(sample.weight, environment_radiance_gpu(scene, sample.direction, settings)), 64.0f);
}

__device__ int irradiance_grid_sample_index_gpu(int resolution, int x, int y, int z) {
    return (z * resolution + y) * resolution + x;
}

__device__ int irradiance_grid_cell_index_gpu(int resolution, int x, int y, int z) {
    const int cells_per_axis = resolution - 1;
    return (z * cells_per_axis + y) * cells_per_axis + x;
}

__device__ bool irradiance_aabb_valid_gpu(Vec3 bounds_min, Vec3 bounds_max) {
    return isfinite(bounds_min.x) && isfinite(bounds_min.y) && isfinite(bounds_min.z) &&
        isfinite(bounds_max.x) && isfinite(bounds_max.y) && isfinite(bounds_max.z) &&
        bounds_max.x > bounds_min.x && bounds_max.y > bounds_min.y && bounds_max.z > bounds_min.z;
}

__device__ int nearest_irradiance_direction_gpu(const GpuIrradianceVolume& volume, Vec3 direction) {
    direction = dnormalize(direction);
    if (ddot(direction, direction) <= 0.0f || volume.direction_count <= 0 || !volume.directions) {
        return -1;
    }
    int best = 0;
    float best_dot = -kInfinity;
    for (int i = 0; i < volume.direction_count; ++i) {
        const float d = ddot(direction, volume.directions[i]);
        if (d > best_dot) {
            best_dot = d;
            best = i;
        }
    }
    return best;
}

__device__ Vec3 irradiance_sample_value_gpu(const GpuIrradianceVolume& volume, const GpuIrradianceVolumeGrid& grid, int direction_index, int x, int y, int z) {
    if (direction_index < 0 || direction_index >= volume.direction_count || !volume.irradiance) {
        return {};
    }
    const int local_sample = irradiance_grid_sample_index_gpu(grid.resolution, x, y, z);
    const int sample_index = grid.sample_offset + local_sample;
    if (sample_index < 0 || sample_index >= volume.irradiance_sample_count) {
        return {};
    }
    return volume.irradiance[sample_index * volume.direction_count + direction_index];
}

__device__ Vec3 query_irradiance_grid_gpu(const GpuIrradianceVolume& volume, const GpuIrradianceVolumeGrid& grid, int direction_index, Vec3 position) {
    if (grid.resolution < 2 || !irradiance_aabb_valid_gpu(grid.bounds_min, grid.bounds_max)) {
        return {};
    }

    const Vec3 extent = sub(grid.bounds_max, grid.bounds_min);
    const float sx = dclamp((position.x - grid.bounds_min.x) / fmaxf(1.0e-8f, extent.x), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const float sy = dclamp((position.y - grid.bounds_min.y) / fmaxf(1.0e-8f, extent.y), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const float sz = dclamp((position.z - grid.bounds_min.z) / fmaxf(1.0e-8f, extent.z), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const int x0 = iclamp_gpu(static_cast<int>(floorf(sx)), 0, grid.resolution - 2);
    const int y0 = iclamp_gpu(static_cast<int>(floorf(sy)), 0, grid.resolution - 2);
    const int z0 = iclamp_gpu(static_cast<int>(floorf(sz)), 0, grid.resolution - 2);
    const float fx = dclamp(sx - static_cast<float>(x0), 0.0f, 1.0f);
    const float fy = dclamp(sy - static_cast<float>(y0), 0.0f, 1.0f);
    const float fz = dclamp(sz - static_cast<float>(z0), 0.0f, 1.0f);

    const Vec3 c000 = irradiance_sample_value_gpu(volume, grid, direction_index, x0, y0, z0);
    const Vec3 c100 = irradiance_sample_value_gpu(volume, grid, direction_index, x0 + 1, y0, z0);
    const Vec3 c010 = irradiance_sample_value_gpu(volume, grid, direction_index, x0, y0 + 1, z0);
    const Vec3 c110 = irradiance_sample_value_gpu(volume, grid, direction_index, x0 + 1, y0 + 1, z0);
    const Vec3 c001 = irradiance_sample_value_gpu(volume, grid, direction_index, x0, y0, z0 + 1);
    const Vec3 c101 = irradiance_sample_value_gpu(volume, grid, direction_index, x0 + 1, y0, z0 + 1);
    const Vec3 c011 = irradiance_sample_value_gpu(volume, grid, direction_index, x0, y0 + 1, z0 + 1);
    const Vec3 c111 = irradiance_sample_value_gpu(volume, grid, direction_index, x0 + 1, y0 + 1, z0 + 1);
    const Vec3 c00 = dlerp(c000, c100, fx);
    const Vec3 c10 = dlerp(c010, c110, fx);
    const Vec3 c01 = dlerp(c001, c101, fx);
    const Vec3 c11 = dlerp(c011, c111, fx);
    const Vec3 c0 = dlerp(c00, c10, fy);
    const Vec3 c1 = dlerp(c01, c11, fy);
    return dlerp(c0, c1, fz);
}

__device__ int query_irradiance_grid_index_gpu(const GpuIrradianceVolume& volume, Vec3 position) {
    if (volume.grid_count <= 0 || !volume.grids) {
        return -1;
    }
    int grid_index = 0;
    for (int depth = 0; depth < 4; ++depth) {
        const GpuIrradianceVolumeGrid grid = volume.grids[grid_index];
        if (grid.resolution < 2 || grid.cell_count <= 0 || !volume.cell_subgrid_indices ||
            !irradiance_aabb_valid_gpu(grid.bounds_min, grid.bounds_max)) {
            return grid_index;
        }
        const Vec3 extent = sub(grid.bounds_max, grid.bounds_min);
        const float sx = dclamp((position.x - grid.bounds_min.x) / fmaxf(1.0e-8f, extent.x), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
        const float sy = dclamp((position.y - grid.bounds_min.y) / fmaxf(1.0e-8f, extent.y), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
        const float sz = dclamp((position.z - grid.bounds_min.z) / fmaxf(1.0e-8f, extent.z), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
        const int x = iclamp_gpu(static_cast<int>(floorf(sx)), 0, grid.resolution - 2);
        const int y = iclamp_gpu(static_cast<int>(floorf(sy)), 0, grid.resolution - 2);
        const int z = iclamp_gpu(static_cast<int>(floorf(sz)), 0, grid.resolution - 2);
        const int local_cell = irradiance_grid_cell_index_gpu(grid.resolution, x, y, z);
        if (local_cell < 0 || local_cell >= grid.cell_count) {
            return grid_index;
        }
        const int subgrid_index = volume.cell_subgrid_indices[grid.cell_offset + local_cell];
        if (subgrid_index < 0 || subgrid_index >= volume.grid_count || subgrid_index == grid_index) {
            return grid_index;
        }
        grid_index = subgrid_index;
    }
    return grid_index;
}

__device__ Vec3 query_irradiance_volume_gpu(const GpuIrradianceVolume& volume, Vec3 position, Vec3 normal) {
    const int direction_index = nearest_irradiance_direction_gpu(volume, normal);
    const int grid_index = query_irradiance_grid_index_gpu(volume, position);
    if (direction_index < 0 || grid_index < 0 || grid_index >= volume.grid_count) {
        return {};
    }
    return query_irradiance_grid_gpu(volume, volume.grids[grid_index], direction_index, position);
}

__device__ Vec3 lightmap_texel_gpu(const GpuLightmap& lightmap, int x, int y) {
    if (lightmap.width <= 0 || lightmap.height <= 0 || !lightmap.texels) {
        return {};
    }
    x = iclamp_gpu(x, 0, lightmap.width - 1);
    y = iclamp_gpu(y, 0, lightmap.height - 1);
    return lightmap.texels[y * lightmap.width + x];
}

__device__ Vec3 query_lightmap_gpu(const GpuLightmap& lightmap, Vec2 uv) {
    if (lightmap.width <= 0 || lightmap.height <= 0 || !lightmap.texels) {
        return {};
    }
    const float sx = dclamp(uv.x, 0.0f, 1.0f) * static_cast<float>(lightmap.width - 1);
    const float sy = dclamp(uv.y, 0.0f, 1.0f) * static_cast<float>(lightmap.height - 1);
    const int x0 = iclamp_gpu(static_cast<int>(floorf(sx)), 0, lightmap.width - 1);
    const int y0 = iclamp_gpu(static_cast<int>(floorf(sy)), 0, lightmap.height - 1);
    const int x1 = x0 + 1 < lightmap.width ? x0 + 1 : lightmap.width - 1;
    const int y1 = y0 + 1 < lightmap.height ? y0 + 1 : lightmap.height - 1;
    const float fx = dclamp(sx - static_cast<float>(x0), 0.0f, 1.0f);
    const float fy = dclamp(sy - static_cast<float>(y0), 0.0f, 1.0f);
    return dlerp(
        dlerp(lightmap_texel_gpu(lightmap, x0, y0), lightmap_texel_gpu(lightmap, x1, y0), fx),
        dlerp(lightmap_texel_gpu(lightmap, x0, y1), lightmap_texel_gpu(lightmap, x1, y1), fx),
        fy);
}

__device__ bool material_uses_irradiance_volume_gi_gpu(const RenderSettings& settings, const GpuMaterial& material) {
    if (material.brdf_model == static_cast<int>(BrdfModel::Lambertian)) {
        return true;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::StandardSurface) && material.transmission > 0.05f) {
        return false;
    }
    return settings.irradiance_volume_principled_gi &&
        (material.brdf_model == static_cast<int>(BrdfModel::Principled) ||
            material.brdf_model == static_cast<int>(BrdfModel::StandardSurface));
}

__device__ bool material_uses_lightmap_gi_gpu(const RenderSettings& settings, const GpuMaterial& material) {
    if (material.brdf_model == static_cast<int>(BrdfModel::Lambertian)) {
        return true;
    }
    if (material.brdf_model == static_cast<int>(BrdfModel::StandardSurface) && material.transmission > 0.05f) {
        return false;
    }
    return settings.lightmap_principled_gi &&
        (material.brdf_model == static_cast<int>(BrdfModel::Principled) ||
            material.brdf_model == static_cast<int>(BrdfModel::StandardSurface));
}

__device__ Vec3 shade_material_from_lightmap_gpu(
    const GpuScene& scene,
    const RenderSettings& settings,
    const GpuMaterial& material,
    const GpuHit& hit,
    Vec3 wo,
    uint32_t& rng) {
    const Vec3 base_color = mul(material.albedo, sample_texture_gpu(scene, material.texture_index, material_base_uv_gpu(material, hit.uv)));
    const float diffuse_scale = material.brdf_model == static_cast<int>(BrdfModel::Principled)
        ? 1.0f - material_metallic_gpu(scene, material, hit.uv)
        : material.brdf_model == static_cast<int>(BrdfModel::StandardSurface)
            ? (1.0f - material_metallic_gpu(scene, material, hit.uv)) * (1.0f - material_transmission_gpu(scene, material, hit.uv))
        : 1.0f;
    const Vec3 irradiance = query_lightmap_gpu(scene.lightmap, hit.lightmap_uv);
    const Vec3 indirect = mul(mul(base_color, diffuse_scale), divv(irradiance, kPi));
    const Vec3 direct_lights = estimate_direct_gpu(scene, hit, material, wo, rng, settings);
    const Vec3 direct_environment = estimate_direct_environment_gpu(scene, hit, material, wo, rng, settings);
    return add(add(direct_lights, direct_environment), indirect);
}

__device__ Vec3 shade_material_from_irradiance_volume_gpu(
    const GpuScene& scene,
    const RenderSettings& settings,
    const GpuMaterial& material,
    const GpuHit& hit,
    Vec3 wo,
    uint32_t& rng) {
    const Vec3 base_color = mul(material.albedo, sample_texture_gpu(scene, material.texture_index, material_base_uv_gpu(material, hit.uv)));
    const float diffuse_scale = material.brdf_model == static_cast<int>(BrdfModel::Principled)
        ? 1.0f - material_metallic_gpu(scene, material, hit.uv)
        : material.brdf_model == static_cast<int>(BrdfModel::StandardSurface)
            ? (1.0f - material_metallic_gpu(scene, material, hit.uv)) * (1.0f - material_transmission_gpu(scene, material, hit.uv))
        : 1.0f;
    const Vec3 irradiance = query_irradiance_volume_gpu(scene.irradiance_volume, hit.position, hit.normal);
    const Vec3 indirect = mul(mul(base_color, diffuse_scale), divv(irradiance, kPi));
    const Vec3 direct_lights = estimate_direct_gpu(scene, hit, material, wo, rng, settings);
    const Vec3 direct_environment = estimate_direct_environment_gpu(scene, hit, material, wo, rng, settings);
    return add(add(direct_lights, direct_environment), indirect);
}

struct GpuIrradianceVolumeProbeHit {
    float t = kInfinity;
    Vec3 position;
    Vec3 normal;
};

__device__ bool intersect_irradiance_debug_probe_sphere_gpu(const GpuIrradianceVolumeDebugProbe& probe, const Ray& ray, float radius_scale, GpuIrradianceVolumeProbeHit& hit) {
    const float radius = fmaxf(1.0e-5f, probe.radius * fmaxf(0.0f, radius_scale));
    const Vec3 oc = sub(ray.origin, probe.position);
    const float a = ddot(ray.direction, ray.direction);
    const float half_b = ddot(oc, ray.direction);
    const float c = ddot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float root = sqrtf(discriminant);
    float t = (-half_b - root) / a;
    if (t <= 0.001f || t >= hit.t) {
        t = (-half_b + root) / a;
        if (t <= 0.001f || t >= hit.t) {
            return false;
        }
    }
    hit.t = t;
    hit.position = add(ray.origin, mul(ray.direction, t));
    hit.normal = dnormalize(divv(sub(hit.position, probe.position), radius));
    return true;
}

__device__ bool intersect_irradiance_debug_probes_gpu(const GpuIrradianceVolume& volume, const Ray& ray, float radius_scale, GpuIrradianceVolumeProbeHit& hit) {
    if (radius_scale <= 0.0f || volume.debug_probe_count <= 0 || !volume.debug_probes) {
        return false;
    }
    bool found = false;
    for (int i = 0; i < volume.debug_probe_count; ++i) {
        found = intersect_irradiance_debug_probe_sphere_gpu(volume.debug_probes[i], ray, radius_scale, hit) || found;
    }
    return found;
}

__device__ Vec3 shade_irradiance_debug_probe_gpu(const GpuIrradianceVolumeProbeHit& hit, Vec3 view_direction) {
    const Vec3 view_light = dnormalize(Vec3{-0.35f, 0.55f, -0.75f});
    const float lambert = fmaxf(0.0f, ddot(hit.normal, mul(view_light, -1.0f)));
    const float rim = powf(dclamp(1.0f - fabsf(ddot(hit.normal, mul(view_direction, -1.0f))), 0.0f, 1.0f), 2.0f);
    const float tone = dclamp(0.28f + lambert * 0.55f + rim * 0.18f, 0.0f, 1.0f);
    return {tone, tone, tone};
}

__device__ Vec3 trace_gpu(const GpuScene& scene, Ray ray, uint32_t& rng, const RenderSettings& settings) {
    Vec3 radiance{};
    Vec3 throughput{1.0f, 1.0f, 1.0f};
    Vec3 previous_position{};
    float previous_bsdf_pdf = 0.0f;
    bool previous_delta = false;
    int transmission_bounces = 0;
    constexpr int kExtraTransmissionBounces = 12;
    for (int bounce = 0; bounce < settings.max_bounces + kExtraTransmissionBounces; ++bounce) {
        const int shading_bounce = bounce - transmission_bounces;
        if (shading_bounce >= settings.max_bounces) {
            break;
        }
        const float sample_clamp = shading_bounce == 0 ? 64.0f : 8.0f;
        GpuHit hit;
        if (!intersect_gpu(scene, ray, hit)) {
            radiance = add(radiance, clamp_sample_radiance_gpu(mul(throughput, environment_radiance_gpu(scene, ray.direction, settings)), sample_clamp));
            break;
        }
        if (hit.material < 0 || hit.material >= scene.material_count) {
            break;
        }
        const GpuMaterial material = scene.materials[hit.material];
        if (!material_visible_gpu(scene, material, hit.uv, rng)) {
            ray = {add(hit.position, mul(ray.direction, 0.002f)), ray.direction};
            --bounce;
            continue;
        }
        hit.normal = apply_normal_map_gpu(scene, material, hit, ray.direction);
        hit.emission = add(hit.emission, material_emission_gpu(scene, material, hit.uv));
        if (has_light_emission_gpu(hit.emission)) {
            const GpuTriangle light = hit.triangle >= 0 && hit.triangle < scene.triangle_count ? scene.triangles[hit.triangle] : GpuTriangle{};
            const GpuMaterial light_material = scene.materials[hit.material];
            const Vec3 emission = emitted_radiance_gpu(light, light_material, light.emission, material_emission_gpu(scene, light_material, hit.uv), light.light_double_sided != 0, ray.direction);
            if (shading_bounce == 0) {
                radiance = add(radiance, clamp_sample_radiance_gpu(mul(throughput, emission), sample_clamp));
            } else if (previous_delta) {
                radiance = add(radiance, clamp_sample_radiance_gpu(mul(throughput, emission), sample_clamp));
            } else if (settings.use_mis && hit.triangle >= 0 && hit.triangle < scene.triangle_count) {
                const float light_pmf = scene.light_count > 0 ? 1.0f / static_cast<float>(scene.light_count) : 0.0f;
                const float light_pdf = light_pdf_solid_angle_gpu(light, light_material, previous_position, hit.position, light_pmf);
                radiance = add(radiance, clamp_sample_radiance_gpu(mul(mul(throughput, emission), mis_weight_gpu(previous_bsdf_pdf, light_pdf, static_cast<int>(settings.mis_heuristic))), sample_clamp));
            }
            break;
        }
        const Vec3 wo = mul(ray.direction, -1.0f);
        if (settings.use_lightmap && hit.has_lightmap && material_uses_lightmap_gi_gpu(settings, material)) {
            radiance = add(radiance, clamp_sample_radiance_gpu(
                mul(throughput, shade_material_from_lightmap_gpu(scene, settings, material, hit, wo, rng)),
                sample_clamp));
            break;
        }
        if (settings.use_irradiance_volume && material_uses_irradiance_volume_gi_gpu(settings, material)) {
            radiance = add(radiance, clamp_sample_radiance_gpu(
                mul(throughput, shade_material_from_irradiance_volume_gpu(scene, settings, material, hit, wo, rng)),
                sample_clamp));
            break;
        }
        radiance = add(radiance, clamp_sample_radiance_gpu(mul(throughput, estimate_direct_gpu(scene, hit, material, wo, rng, settings)), sample_clamp));
        if (shading_bounce >= 3) {
            const float p = dclamp(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng_float(rng) > p) {
                break;
            }
            throughput = divv(throughput, p);
        }
        const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal, wo, hit.uv, hit.front_face, rng);
        if (!isfinite(sample.pdf) || sample.pdf <= 0.0f || !finite_vec_gpu(sample.weight) || ddot(sample.weight, sample.weight) <= 0.0f) {
            break;
        }
        previous_position = hit.position;
        previous_bsdf_pdf = sample.pdf;
        previous_delta = sample.delta;
        if (sample.delta && material_transmission_gpu(scene, material, hit.uv) > 0.5f && transmission_bounces < kExtraTransmissionBounces) {
            ++transmission_bounces;
        }
        const float offset_side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
        ray = {add(hit.position, mul(hit.normal, 0.001f * offset_side)), sample.direction};
        throughput = mul(throughput, sample.weight);
    }
    return radiance;
}

__device__ Vec3 trace_gpu_with_irradiance_probe_debug(const GpuScene& scene, Ray ray, uint32_t& rng, const RenderSettings& settings) {
    if (!settings.use_irradiance_volume || !settings.irradiance_volume_debug_probes ||
        scene.irradiance_volume.debug_probe_count <= 0) {
        return trace_gpu(scene, ray, rng, settings);
    }

    GpuIrradianceVolumeProbeHit probe_hit;
    if (!intersect_irradiance_debug_probes_gpu(scene.irradiance_volume, ray, settings.irradiance_volume_debug_probe_radius_scale, probe_hit)) {
        return trace_gpu(scene, ray, rng, settings);
    }

    GpuHit scene_hit;
    if (intersect_gpu(scene, ray, scene_hit) && scene_hit.t < probe_hit.t) {
        return trace_gpu(scene, ray, rng, settings);
    }
    return shade_irradiance_debug_probe_gpu(probe_hit, ray.direction);
}

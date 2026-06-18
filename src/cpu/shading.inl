Vec3 light_emission(const Scene& scene, int mesh_index) {
    if (mesh_index < 0 || mesh_index >= static_cast<int>(scene.meshes.size())) {
        return {};
    }
    const LightComponent& light = scene.meshes[static_cast<size_t>(mesh_index)].light;
    return light.enabled && light.intensity > 0.0f ? light.color * light.intensity : Vec3{};
}

Vec3 material_emission(const Scene& scene, int material_index, Vec2 uv) {
    if (material_index < 0 || material_index >= static_cast<int>(scene.materials.size()) || !scene.materials[static_cast<size_t>(material_index)]) {
        return {};
    }
    return scene.materials[static_cast<size_t>(material_index)]->emitted(uv);
}

bool has_light_emission(Vec3 emission) {
    return emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f;
}

bool is_finite(Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

Vec3 clamp_sample_radiance(Vec3 v, float limit = 64.0f) {
    if (!is_finite(v)) {
        return {};
    }
    const float max_sample_radiance = std::max(0.0f, limit);
    return {
        std::clamp(v.x, 0.0f, max_sample_radiance),
        std::clamp(v.y, 0.0f, max_sample_radiance),
        std::clamp(v.z, 0.0f, max_sample_radiance),
    };
}

bool mesh_light_is_enabled(const Scene& scene, int mesh_index) {
    return mesh_index >= 0 && mesh_index < static_cast<int>(scene.meshes.size()) &&
        scene.meshes[static_cast<size_t>(mesh_index)].light.enabled &&
        scene.meshes[static_cast<size_t>(mesh_index)].light.intensity > 0.0f;
}

bool mesh_light_is_double_sided(const Scene& scene, int mesh_index) {
    return mesh_light_is_enabled(scene, mesh_index) && scene.meshes[static_cast<size_t>(mesh_index)].light.double_sided;
}

Vec3 emitted_radiance(const Scene& scene, const Triangle& light, Vec2 uv, Vec3 ray_direction_to_light) {
    const Vec3 mesh_emission = light_emission(scene, light.mesh);
    const Vec3 mat_emission = material_emission(scene, light.material, uv);
    const bool front_facing = dot(light.normal, -ray_direction_to_light) > 0.0f;
    Vec3 emission;
    if (has_light_emission(mesh_emission) && (mesh_light_is_double_sided(scene, light.mesh) || front_facing)) {
        emission += mesh_emission;
    }
    const bool material_double_sided = light.material >= 0 && light.material < static_cast<int>(scene.materials.size()) &&
        scene.materials[static_cast<size_t>(light.material)] && scene.materials[static_cast<size_t>(light.material)]->double_sided;
    if (has_light_emission(mat_emission) && (material_double_sided || front_facing)) {
        emission += mat_emission;
    }
    return emission;
}

Vec3 apply_normal_map(const Material& material, const Hit& hit, Vec3 ray_direction) {
    if (!material.normal_texture) {
        return hit.normal;
    }
    const Vec3 sample = material.normal_texture->sample(hit.uv) * 2.0f - Vec3{1.0f};
    const Vec3 tangent = normalize(hit.tangent - hit.normal * dot(hit.normal, hit.tangent));
    const Vec3 bitangent = dot(hit.bitangent, hit.bitangent) > 0.0f ? hit.bitangent : cross(hit.normal, tangent);
    Vec3 mapped = normalize(tangent * (sample.x * material.normal_scale) + bitangent * (sample.y * material.normal_scale) + hit.normal * sample.z);
    if (dot(mapped, mapped) <= 0.0f) {
        mapped = hit.normal;
    }
    return face_forward(mapped, ray_direction);
}

bool material_visible(const Material& material, Vec2 uv, Rng& rng) {
    const float opacity = material.opacity(uv);
    if (material.alpha_mode == AlphaMode::Opaque) {
        return true;
    }
    if (material.alpha_mode == AlphaMode::Mask) {
        return opacity >= material.alpha_cutoff;
    }
    return rng.next_float() <= opacity;
}

Vec2 equal_area_sphere_to_square(Vec3 direction) {
    const float ax = std::fabs(direction.x);
    const float ay = std::fabs(direction.y);
    const float az = std::fabs(direction.z);
    const float r = std::sqrt(std::max(0.0f, 1.0f - az));
    if (r <= 1.0e-8f) {
        return {0.5f, direction.z >= 0.0f ? 0.5f : 0.0f};
    }
    const float phi = std::atan2(ay, ax);
    const float a = 4.0f * phi / kPi - 1.0f;
    const float sum = direction.z >= 0.0f ? r : 2.0f - r;
    const float diff = a * r;
    const float up = (sum - diff) * 0.5f;
    const float vp = (sum + diff) * 0.5f;
    const float u = direction.x < 0.0f ? -up : up;
    const float v = direction.y < 0.0f ? -vp : vp;
    return {u * 0.5f + 0.5f, v * 0.5f + 0.5f};
}

float environment_mip_lod(const Scene& scene, const RenderSettings& settings) {
    if (!scene.environment.texture || scene.environment.texture->width <= 0 || scene.environment.texture->height <= 0 || settings.width <= 0 || settings.height <= 0) {
        return 0.0f;
    }
    const float aspect = static_cast<float>(settings.width) / static_cast<float>(settings.height);
    const float half_height = std::tan(scene.camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const float ray_solid_angle = std::max(1.0e-8f, (4.0f * half_width * half_height) / static_cast<float>(settings.width * settings.height));
    const float texel_solid_angle = 4.0f * kPi / static_cast<float>(scene.environment.texture->width * scene.environment.texture->height);
    return std::max(0.0f, 0.5f * std::log2(ray_solid_angle / std::max(1.0e-12f, texel_solid_angle)));
}

Vec3 environment_radiance(const Scene& scene, Vec3 direction, const RenderSettings& settings) {
    if (scene.environment.texture) {
        direction = normalize({
            dot(direction, scene.environment.light_from_world_x),
            dot(direction, scene.environment.light_from_world_y),
            dot(direction, scene.environment.light_from_world_z),
        });
        Vec2 uv;
        if (scene.environment.mapping == Environment::Mapping::EqualArea) {
            uv = equal_area_sphere_to_square(direction);
        } else {
            const float u = std::atan2(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
            const float v = std::acos(std::clamp(direction.y, -1.0f, 1.0f)) / kPi;
            uv = {u, 1.0f - v};
        }
        const TextureWrap2D wrap = scene.environment.mapping == Environment::Mapping::EqualArea ? TextureWrap2D::OctahedralSphere : TextureWrap2D::RepeatClampY;
        return scene.environment.color * scene.environment.texture->sample_lod(uv, environment_mip_lod(scene, settings), wrap) * scene.environment.strength;
    }
    if (scene.environment.constant) {
        return scene.environment.color * scene.environment.strength;
    }
    const float t = 0.5f * (direction.y + 1.0f);
    return ((1.0f - t) * Vec3{0.02f, 0.025f, 0.035f} + t * Vec3{0.32f, 0.45f, 0.68f}) * scene.environment.color * scene.environment.strength;
}

Vec3 sample_triangle_area(const Triangle& tri, Rng& rng) {
    const float su = std::sqrt(rng.next_float());
    const float v = rng.next_float();
    return tri.v0 * (1.0f - su) + tri.v1 * (su * (1.0f - v)) + tri.v2 * (su * v);
}

Vec2 sample_triangle_uv(const Triangle& tri, Rng& rng, Vec3& point) {
    const float su = std::sqrt(rng.next_float());
    const float v = rng.next_float();
    const float b0 = 1.0f - su;
    const float b1 = su * (1.0f - v);
    const float b2 = su * v;
    point = tri.v0 * b0 + tri.v1 * b1 + tri.v2 * b2;
    return tri.uv0 * b0 + tri.uv1 * b1 + tri.uv2 * b2;
}

float triangle_area(const Triangle& tri) {
    return 0.5f * length(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

float light_pdf_solid_angle(const Scene& scene, const Triangle& light, Vec3 origin, Vec3 light_point, float light_pmf) {
    const Vec3 to_light = light_point - origin;
    const float dist2 = dot(to_light, to_light);
    if (dist2 <= 1.0e-8f) {
        return 0.0f;
    }
    const Vec3 light_dir = to_light / std::sqrt(dist2);
    const bool has_mesh_emission = has_light_emission(light_emission(scene, light.mesh));
    const bool material_emissive_double_sided = !has_mesh_emission &&
        light.material >= 0 && light.material < static_cast<int>(scene.materials.size()) &&
        scene.materials[static_cast<size_t>(light.material)] &&
        scene.materials[static_cast<size_t>(light.material)]->double_sided &&
        has_light_emission(scene.materials[static_cast<size_t>(light.material)]->emitted({0.0f, 0.0f}));
    const bool double_sided = mesh_light_is_double_sided(scene, light.mesh) || material_emissive_double_sided;
    const float ldot_raw = dot(-light.normal, light_dir);
    const float ldot = double_sided ? std::fabs(ldot_raw) : std::max(0.0f, ldot_raw);
    const float area = triangle_area(light);
    return ldot > 0.0f && area > 0.0f && light_pmf > 0.0f ? light_pmf * dist2 / (ldot * area) : 0.0f;
}

float mis_weight(float pdf_a, float pdf_b, MisHeuristic heuristic) {
    if (heuristic == MisHeuristic::Balance) {
        return pdf_a / std::max(1.0e-8f, pdf_a + pdf_b);
    }
    const float a = pdf_a;
    const float b = pdf_b;
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 / std::max(1.0e-8f, a2 + b2);
}

Vec3 estimate_direct_lighting(const RenderScene& render_scene, const Scene& scene, const Hit& hit, const Material& material, Vec3 wo, Rng& rng, const RenderSettings& settings) {
    Vec3 direct;
    const int light_count = static_cast<int>(render_scene.light_triangle_indices.size());
    if (light_count <= 0) {
        return direct;
    }
    const int light_list_index = std::min(static_cast<int>(rng.next_float() * static_cast<float>(light_count)), light_count - 1);
    const int light_index = render_scene.light_triangle_indices[static_cast<size_t>(light_list_index)];
    if (light_index < 0 || light_index >= static_cast<int>(render_scene.triangles.size())) {
        return direct;
    }
    const Triangle& light = render_scene.triangles[static_cast<size_t>(light_index)];

    Vec3 light_point;
    const Vec2 light_uv = sample_triangle_uv(light, rng, light_point);
    const Vec3 to_light = light_point - hit.position;
    const float dist2 = dot(to_light, to_light);
    if (dist2 <= 1.0e-8f) {
        return direct;
    }
    const float dist = std::sqrt(dist2);
    const Vec3 light_dir = to_light / dist;
    const float ndotl_raw = dot(hit.normal, light_dir);
    const float ndotl = material.double_sided ? std::fabs(ndotl_raw) : std::max(0.0f, ndotl_raw);
    const bool has_mesh_emission = has_light_emission(light_emission(scene, light.mesh));
    const bool material_emissive_double_sided = !has_mesh_emission &&
        light.material >= 0 && light.material < static_cast<int>(scene.materials.size()) &&
        scene.materials[static_cast<size_t>(light.material)] &&
        scene.materials[static_cast<size_t>(light.material)]->double_sided &&
        has_light_emission(scene.materials[static_cast<size_t>(light.material)]->emitted(light_uv));
    const bool light_double_sided = mesh_light_is_double_sided(scene, light.mesh) || material_emissive_double_sided;
    const float ldot_raw = dot(-light.normal, light_dir);
    const float ldot = light_double_sided ? std::fabs(ldot_raw) : std::max(0.0f, ldot_raw);
    if (ndotl <= 0.0f || ldot <= 0.0f) {
        return direct;
    }

    const float shadow_offset_side = ndotl_raw >= 0.0f ? 1.0f : -1.0f;
    Ray shadow_ray{hit.position + hit.normal * (0.002f * shadow_offset_side), light_dir};
    for (int shadow_step = 0; shadow_step < 8; ++shadow_step) {
        Hit shadow_hit;
        if (!intersect_scene(render_scene, shadow_ray, shadow_hit, settings.acceleration_structure)) {
            break;
        }
        const float shadow_dist = length(shadow_hit.position - hit.position);
        if (shadow_dist >= dist - 0.01f || shadow_hit.triangle == light_index) {
            break;
        }
        if (shadow_hit.material >= 0 && shadow_hit.material < static_cast<int>(scene.materials.size()) &&
            scene.materials[static_cast<size_t>(shadow_hit.material)]) {
            const Material& shadow_material = *scene.materials[static_cast<size_t>(shadow_hit.material)];
            if (!material_visible(shadow_material, shadow_hit.uv, rng) || shadow_material.model() == BrdfModel::Dielectric) {
                shadow_ray = {shadow_hit.position + light_dir * 0.002f, light_dir};
                continue;
            }
        }
        return direct;
    }

    const float light_pmf = 1.0f / static_cast<float>(light_count);
    const float light_pdf = light_pdf_solid_angle(scene, light, hit.position, light_point, light_pmf);
    if (!std::isfinite(light_pdf) || light_pdf <= 0.0f) {
        return direct;
    }
    const Vec3 emission = emitted_radiance(scene, light, light_uv, light_dir);
    if (!has_light_emission(emission)) {
        return direct;
    }
    const float bsdf_pdf = material.pdf(hit.normal, wo, light_dir, hit.uv);
    if (!std::isfinite(bsdf_pdf) || bsdf_pdf < 0.0f) {
        return direct;
    }
    const float weight = settings.use_mis ? mis_weight(light_pdf, bsdf_pdf, settings.mis_heuristic) : 1.0f;
    direct += clamp_sample_radiance(material.evaluate(hit.normal, wo, light_dir, hit.uv) * emission * (ndotl / light_pdf) * weight);
    return direct;
}

Vec3 trace_path(const RenderScene& render_scene, const Scene& scene, Ray ray, Rng& rng, const RenderSettings& settings) {
    Vec3 radiance;
    Vec3 throughput{1.0f};
    Vec3 previous_position;
    float previous_bsdf_pdf = 0.0f;
    bool previous_delta = false;

    for (int bounce = 0; bounce < settings.max_bounces; ++bounce) {
        const float sample_clamp = bounce == 0 ? 64.0f : 8.0f;
        Hit hit;
        if (!intersect_scene(render_scene, ray, hit, settings.acceleration_structure)) {
            radiance += clamp_sample_radiance(throughput * environment_radiance(scene, ray.direction, settings), sample_clamp);
            break;
        }

        if (hit.material < 0 || hit.material >= static_cast<int>(scene.materials.size())) {
            break;
        }
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(hit.material)];
        if (!material) {
            break;
        }
        if (!material_visible(*material, hit.uv, rng)) {
            ray = {hit.position + ray.direction * 0.002f, ray.direction};
            --bounce;
            continue;
        }
        hit.normal = apply_normal_map(*material, hit, ray.direction);
        Vec3 emission;
        if (hit.triangle >= 0 && hit.triangle < static_cast<int>(render_scene.triangles.size())) {
            emission = emitted_radiance(scene, render_scene.triangles[static_cast<size_t>(hit.triangle)], hit.uv, ray.direction);
        }
        if (has_light_emission(emission)) {
            if (bounce == 0) {
                radiance += clamp_sample_radiance(throughput * emission, sample_clamp);
            } else if (previous_delta) {
                radiance += clamp_sample_radiance(throughput * emission, sample_clamp);
            } else if (settings.use_mis && hit.triangle >= 0 && hit.triangle < static_cast<int>(render_scene.triangles.size())) {
                const Triangle& light = render_scene.triangles[static_cast<size_t>(hit.triangle)];
                const float light_pmf = render_scene.light_triangle_indices.empty() ? 0.0f : 1.0f / static_cast<float>(render_scene.light_triangle_indices.size());
                const float light_pdf = light_pdf_solid_angle(scene, light, previous_position, hit.position, light_pmf);
                radiance += clamp_sample_radiance(throughput * emission * mis_weight(previous_bsdf_pdf, light_pdf, settings.mis_heuristic), sample_clamp);
            }
            break;
        }
        const Vec3 wo = -ray.direction;
        radiance += clamp_sample_radiance(throughput * estimate_direct_lighting(render_scene, scene, hit, *material, wo, rng, settings), sample_clamp);

        if (bounce >= 3) {
            const float p = std::clamp(std::max({throughput.x, throughput.y, throughput.z}), 0.05f, 0.95f);
            if (rng.next_float() > p) {
                break;
            }
            throughput = throughput / p;
        }

        const MaterialSample sample = material->sample(hit.normal, wo, hit.uv, hit.front_face, rng);
        if (!std::isfinite(sample.pdf) || sample.pdf <= 0.0f || !is_finite(sample.weight) || dot(sample.weight, sample.weight) <= 0.0f) {
            break;
        }
        previous_position = hit.position;
        previous_bsdf_pdf = sample.pdf;
        previous_delta = sample.delta;
        const float offset_side = dot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
        ray = {hit.position + hit.normal * (0.001f * offset_side), sample.direction};
        throughput = throughput * sample.weight;
    }
    return radiance;
}

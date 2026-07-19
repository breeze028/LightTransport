struct GpuRestirGiInitialSample {
    Ray ray = {};
    GpuRestirSurface primary_surface = {};
    Vec3 sample_weight = {};
    float sample_pdf = 0.0f;
    int path_index = -1;
    int valid = 0;
};

struct GpuRestirGiReservoir {
    Vec3 position = {};
    Vec3 normal = {};
    Vec3 radiance = {};
    float weight_sum = 0.0f;
    int M = 0;
    int age = 0;
    int primitive_type = 0;
    int primitive_index = -1;
};

struct GpuRestirGiTemporalState {
    int previous_pixel = -1;
    int previous_m = 0;
    int selected_previous = 0;
    uint32_t correction_rng = 0;
};

struct GpuRestirGiSpatialState {
    int neighbor_pixels[2] = {-1, -1};
    int neighbor_m[2] = {};
    int neighbor_count = 0;
    int selected_neighbor = -1;
    uint32_t correction_rng = 0;
};

// The persistent representation is 32 bytes, matching RTXDI's GI reservoir footprint.
// Radiance uses RTXDI's 14-bit log luminance and 9-bit CIE u/v chroma encoding.
struct GpuPackedRestirGiReservoir {
    Vec3 position = {};
    uint32_t packed_age_m = 0;
    uint32_t packed_radiance = 0;
    float weight_sum = 0.0f;
    uint32_t packed_normal = 0;
    uint32_t unused = 0;
};

static_assert(sizeof(GpuPackedRestirGiReservoir) == 32, "packed ReSTIR GI reservoir must be 32 bytes");

static constexpr int kRestirGiMaxHistoryLength = 8;
static constexpr int kRestirGiMaxAge = 30;
static constexpr int kRestirGiSpatialSamples = 2;
static constexpr float kRestirGiDepthThreshold = 0.10f;
static constexpr float kRestirGiNormalThreshold = 0.60f;
static constexpr float kRestirGiSpatialRadius = 32.0f;
static constexpr float kRestirGiBoilingStrength = 0.20f;
static constexpr float kRestirGiFinalMisRoughness = 0.30f;

__device__ bool restir_gi_enabled_gpu(const RenderSettings& settings) {
    return settings.cuda_restir_gi && settings.cuda_wavefront && settings.max_bounces >= 2 &&
        settings.sampling_mode != PathSamplingMode::Unidirectional;
}

__device__ GpuMaterial restir_gi_roughened_material_gpu(const GpuScene& scene,
    GpuMaterial material, Vec2 uv, float roughness_floor) {
    material.roughness = fmaxf(material_roughness_gpu(scene, material, uv), roughness_floor);
    material.metallic = material_metallic_gpu(scene, material, uv);
    material.roughness_texture_index = -1;
    material.metallic_roughness_texture_index = -1;
    material.metallic_texture_index = -1;
    return material;
}

__device__ uint32_t restir_gi_pack_snorm2(Vec3 n) {
    n = dnormalize(n);
    const float inv_l1 = 1.0f / fmaxf(1.0e-8f, fabsf(n.x) + fabsf(n.y) + fabsf(n.z));
    float x = n.x * inv_l1;
    float y = n.y * inv_l1;
    if (n.z < 0.0f) {
        const float old_x = x;
        x = copysignf(1.0f - fabsf(y), old_x);
        y = copysignf(1.0f - fabsf(old_x), y);
    }
    const int ix = static_cast<int>(dclamp(x, -1.0f, 1.0f) * 32767.0f);
    const int iy = static_cast<int>(dclamp(y, -1.0f, 1.0f) * 32767.0f);
    return static_cast<uint32_t>(static_cast<uint16_t>(ix)) |
        (static_cast<uint32_t>(static_cast<uint16_t>(iy)) << 16u);
}

__device__ Vec3 restir_gi_unpack_snorm2(uint32_t packed) {
    const float x = static_cast<float>(static_cast<int16_t>(packed & 0xffffu)) / 32767.0f;
    const float y = static_cast<float>(static_cast<int16_t>(packed >> 16u)) / 32767.0f;
    Vec3 n{x, y, 1.0f - fabsf(x) - fabsf(y)};
    if (n.z < 0.0f) {
        const float old_x = n.x;
        n.x = copysignf(1.0f - fabsf(n.y), old_x);
        n.y = copysignf(1.0f - fabsf(old_x), n.y);
    }
    return dnormalize(n);
}

__device__ uint32_t restir_gi_pack_radiance(Vec3 value) {
    value = {fmaxf(0.0f, value.x), fmaxf(0.0f, value.y), fmaxf(0.0f, value.z)};
    if (!finite_vec_gpu(value)) return 0u;
    const float X = 0.4123908f * value.x + 0.3575844f * value.y + 0.1804808f * value.z;
    const float Y = 0.2126390f * value.x + 0.7151687f * value.y + 0.0721923f * value.z;
    const float Z = 0.0193308f * value.x + 0.1191948f * value.y + 0.9505322f * value.z;
    if (!(Y > 0.0f)) return 0u;
    const uint32_t log_y = static_cast<uint32_t>(dclamp(409.6f * (log2f(Y) + 20.0f), 0.0f, 16383.0f));
    if (log_y == 0u) return 0u;
    const float denominator = -2.0f * X + 12.0f * Y + 3.0f * (X + Y + Z);
    if (!(denominator > 0.0f)) return 0u;
    const uint32_t u = static_cast<uint32_t>(dclamp(820.0f * 4.0f * X / denominator, 0.0f, 511.0f));
    const uint32_t v = static_cast<uint32_t>(dclamp(820.0f * 9.0f * Y / denominator, 0.0f, 511.0f));
    return (log_y << 18u) | (u << 9u) | v;
}

__device__ Vec3 restir_gi_unpack_radiance(uint32_t packed) {
    if (packed == 0u) return {};
    const uint32_t log_y = packed >> 18u;
    if (log_y == 0u) return {};
    const float Y = exp2f((static_cast<float>(log_y) + 0.5f) / 409.6f - 20.0f);
    const float u = (static_cast<float>((packed >> 9u) & 0x1ffu) + 0.5f) / 820.0f;
    const float v = (static_cast<float>(packed & 0x1ffu) + 0.5f) / 820.0f;
    const float inv_denominator = 1.0f / fmaxf(1.0e-8f, 6.0f * u - 16.0f * v + 12.0f);
    const float x = 9.0f * u * inv_denominator;
    const float y = 4.0f * v * inv_denominator;
    const float scale = Y / fmaxf(1.0e-8f, y);
    const float X = scale * x;
    const float Z = scale * (1.0f - x - y);
    return {
        fmaxf(0.0f, 3.2409700f * X - 1.5373832f * Y - 0.4986108f * Z),
        fmaxf(0.0f, -0.9692436f * X + 1.8759675f * Y + 0.0415551f * Z),
        fmaxf(0.0f, 0.0556301f * X - 0.2039769f * Y + 1.0569715f * Z),
    };
}

__device__ GpuPackedRestirGiReservoir restir_gi_pack(const GpuRestirGiReservoir& reservoir) {
    GpuPackedRestirGiReservoir packed;
    if (reservoir.M <= 0) return packed;
    packed.position = reservoir.position;
    packed.packed_age_m = (static_cast<uint32_t>(iclamp_gpu(reservoir.age, 0, 255)) << 8u) |
        static_cast<uint32_t>(iclamp_gpu(reservoir.M, 0, 255));
    packed.packed_radiance = restir_gi_pack_radiance(reservoir.radiance);
    packed.weight_sum = reservoir.weight_sum;
    packed.packed_normal = restir_gi_pack_snorm2(reservoir.normal);
    packed.unused = reservoir.primitive_index >= 0
        ? (static_cast<uint32_t>(reservoir.primitive_type != 0 ? 0x80000000u : 0u) |
            static_cast<uint32_t>(reservoir.primitive_index + 1)) : 0u;
    return packed;
}

__device__ GpuRestirGiReservoir restir_gi_unpack(const GpuPackedRestirGiReservoir& packed) {
    GpuRestirGiReservoir reservoir = {};
    reservoir.M = static_cast<int>(packed.packed_age_m & 0xffu);
    if (reservoir.M <= 0) return {};
    reservoir.age = static_cast<int>((packed.packed_age_m >> 8u) & 0xffu);
    reservoir.position = packed.position;
    reservoir.normal = restir_gi_unpack_snorm2(packed.packed_normal);
    reservoir.radiance = restir_gi_unpack_radiance(packed.packed_radiance);
    reservoir.weight_sum = packed.weight_sum;
    if (packed.unused != 0u) {
        reservoir.primitive_type = (packed.unused & 0x80000000u) != 0u ? 1 : 0;
        reservoir.primitive_index = static_cast<int>((packed.unused & 0x7fffffffu) - 1u);
    }
    return reservoir;
}

__device__ bool restir_gi_valid(const GpuRestirGiReservoir& reservoir) {
    return reservoir.M > 0 && isfinite(reservoir.weight_sum) && finite_vec_gpu(reservoir.radiance);
}

__device__ Vec3 restir_gi_sample_geo_normal_gpu(const GpuScene& scene, const GpuRestirGiReservoir& sample) {
    Vec3 normal = sample.normal;
    if (sample.primitive_type != 0) {
        if (sample.primitive_index >= 0 && sample.primitive_index < scene.sphere_count) {
            normal = dnormalize(sub(sample.position, scene.spheres[sample.primitive_index].center));
        }
    } else if (sample.primitive_index >= 0 && sample.primitive_index < scene.triangle_count) {
        normal = scene.triangles[sample.primitive_index].normal;
    }
    if (!(ddot(normal, normal) > 0.0f)) normal = sample.normal;
    normal = dnormalize(normal);
    return ddot(normal, sample.normal) >= 0.0f ? normal : mul(normal, -1.0f);
}

__device__ bool restir_gi_receiver_accepts_direction_gpu(const GpuRestirSurface& surface,
    const GpuMaterial& material, Vec3 wi) {
    if (material.double_sided || material.brdf_model == static_cast<int>(BrdfModel::DiffuseTransmission)) {
        return true;
    }
    const Vec3 geo_normal = ddot(surface.geo_normal, surface.geo_normal) > 0.0f
        ? surface.geo_normal : surface.normal;
    return ddot(geo_normal, wi) > 0.0f;
}

__device__ bool restir_gi_sample_faces_receiver_gpu(const GpuScene& scene,
    const GpuRestirGiReservoir& sample, Vec3 receiver_position) {
    const Vec3 to_receiver = sub(receiver_position, sample.position);
    const float distance2 = ddot(to_receiver, to_receiver);
    if (!(distance2 > 1.0e-10f)) return false;
    return ddot(restir_gi_sample_geo_normal_gpu(scene, sample), mul(to_receiver, rsqrtf(distance2))) > 0.0f;
}

__device__ float restir_gi_target_gpu(const GpuScene& scene, const GpuRestirSurface& surface,
    const GpuRestirGiReservoir& sample) {
    if (!surface.valid || surface.material < 0 || surface.material >= scene.material_count || !restir_gi_valid(sample)) return 0.0f;
    const Vec3 to_sample = sub(sample.position, surface.position);
    const float distance2 = ddot(to_sample, to_sample);
    if (!(distance2 > 1.0e-10f)) return 0.0f;
    const Vec3 wi = mul(to_sample, rsqrtf(distance2));
    const GpuMaterial material = scene.materials[surface.material];
    if (!restir_gi_receiver_accepts_direction_gpu(surface, material, wi)) return 0.0f;
    const float cosine = (material.double_sided || material.brdf_model == static_cast<int>(BrdfModel::DiffuseTransmission))
        ? fabsf(ddot(surface.normal, wi)) : fmaxf(0.0f, ddot(surface.normal, wi));
    return restir_luminance_gpu(mul(mul(evaluate_brdf_gpu(scene, material, surface.normal,
        surface.wo, wi, surface.uv), sample.radiance), cosine));
}

__device__ float restir_gi_jacobian_gpu(Vec3 receiver, Vec3 source_receiver,
    Vec3 sample_position, Vec3 sample_normal) {
    const Vec3 new_vector = sub(receiver, sample_position);
    const Vec3 old_vector = sub(source_receiver, sample_position);
    const float new_distance2 = ddot(new_vector, new_vector);
    const float old_distance2 = ddot(old_vector, old_vector);
    if (!(new_distance2 > 1.0e-10f) || !(old_distance2 > 1.0e-10f)) return 0.0f;
    const float new_cosine = fmaxf(0.0f, ddot(sample_normal, mul(new_vector, rsqrtf(new_distance2))));
    const float old_cosine = fmaxf(0.0f, ddot(sample_normal, mul(old_vector, rsqrtf(old_distance2))));
    const float jacobian = new_cosine * old_distance2 / fmaxf(1.0e-10f, old_cosine * new_distance2);
    if (!isfinite(jacobian) || jacobian < 0.1f || jacobian > 10.0f) return 0.0f;
    return dclamp(jacobian, 1.0f / 3.0f, 3.0f);
}

__device__ bool restir_gi_surfaces_compatible_gpu(const GpuRestirSurface& current,
    const GpuRestirSurface& other, float expected_depth) {
    if (!current.valid || !other.valid || current.object_id != other.object_id ||
        current.material != other.material) return false;
    if (ddot(current.normal, other.normal) < kRestirGiNormalThreshold) return false;
    if (ddot(current.geo_normal, other.geo_normal) < kRestirGiNormalThreshold) return false;
    return fabsf(other.depth - expected_depth) <= fmaxf(0.01f, fabsf(expected_depth) * kRestirGiDepthThreshold);
}

__device__ int2 restir_gi_temporal_offset_gpu(int sample_index) {
    sample_index &= 7;
    const int mask2 = (sample_index >> 1) & 1;
    const int mask4 = 1 - ((sample_index >> 2) & 1);
    const int tmp0 = -1 + 2 * (sample_index & 1);
    const int tmp1 = 1 - 2 * mask2;
    const int tmp2 = mask4 | mask2;
    const int tmp3 = mask4 | (1 - mask2);
    return make_int2(tmp0 * tmp2, tmp0 * tmp1 * tmp3);
}

__device__ bool restir_gi_combine_gpu(GpuRestirGiReservoir& reservoir,
    const GpuRestirGiReservoir& source, float target, float jacobian, float random) {
    if (!restir_gi_valid(source)) return false;
    const float ris_weight = target * jacobian * source.weight_sum * static_cast<float>(source.M);
    reservoir.M += source.M;
    if (!(ris_weight > 0.0f) || !isfinite(ris_weight)) return false;
    reservoir.weight_sum += ris_weight;
    if (random * reservoir.weight_sum <= ris_weight) {
        reservoir.position = source.position;
        reservoir.normal = source.normal;
        reservoir.radiance = source.radiance;
        reservoir.age = source.age;
        reservoir.primitive_type = source.primitive_type;
        reservoir.primitive_index = source.primitive_index;
        return true;
    }
    return false;
}

__device__ void restir_gi_finalize_gpu(GpuRestirGiReservoir& reservoir,
    float numerator, float denominator) {
    reservoir.weight_sum = denominator > 0.0f ? reservoir.weight_sum * numerator / denominator : 0.0f;
    if (!isfinite(reservoir.weight_sum)) reservoir.weight_sum = 0.0f;
}

__device__ bool restir_gi_visible_gpu(const GpuScene& scene, Vec3 origin, Vec3 normal,
    Vec3 destination, uint32_t& rng) {
    const Vec3 delta = sub(destination, origin);
    const float distance = sqrtf(ddot(delta, delta));
    if (!(distance > 0.003f)) return true;
    const Vec3 direction = divv(delta, distance);
    const float side = ddot(normal, direction) >= 0.0f ? 1.0f : -1.0f;
    Ray ray{add(origin, mul(normal, 0.002f * side)), direction};
    float remaining = distance - 0.01f;
    for (int step = 0; step < 8 && remaining > 0.001f; ++step) {
        GpuHit hit;
        hit.t = remaining;
        if (!intersect_gpu(scene, ray, hit)) return true;
        const GpuMaterial material = scene.materials[hit.material];
        if (!material_visible_gpu(scene, material, hit.uv, rng) ||
            material.brdf_model == static_cast<int>(BrdfModel::Dielectric) ||
            material_transmission_gpu(scene, material, hit.uv) > 0.5f) {
            ray = {add(hit.position, mul(direction, 0.002f)), direction};
            remaining -= hit.t + 0.002f;
            continue;
        }
        return false;
    }
    return true;
}

template <bool HasLocalLights, bool HasEnvironment>
__device__ GpuRestirReservoir restir_gi_sample_secondary_direct_gpu(const GpuScene& scene,
    RenderSettings settings, const GpuHit& hit, const GpuMaterial& material, Vec3 wo, uint32_t& rng) {
    GpuRestirReservoir result;
    restir_clear_gpu(result);
    GpuRestirInitialReservoir strategy;
    int triangle_count = HasLocalLights && scene.light_count > 0 ? 4 : 0;
    int directional_count = HasLocalLights && scene.directional_light_count > 0 ? 1 : 0;
    int point_count = HasLocalLights && scene.point_light_count > 0 ? 1 : 0;
    int environment_count = HasEnvironment ? 2 : 0;
    const int total = triangle_count + directional_count + point_count + environment_count;
    if (total == 0) return result;

    restir_clear_initial_gpu(strategy);
    for (int i = 0; i < triangle_count; ++i) {
        GpuRestirLightSample sample; float discrete_pdf = 0.0f;
        if (!restir_sample_triangle_gpu(scene, rng, sample, discrete_pdf)) continue;
        Vec3 direction; float distance;
        const Vec3 c = restir_evaluate_triangle_sample_gpu(
            scene, hit, material, wo, sample, settings, direction, distance);
        restir_add_initial_gpu(strategy, sample, restir_luminance_gpu(c),
            restir_triangle_proposal_pdf_gpu(scene, hit, sample), rng);
    }
    restir_finalize_initial_strategy_gpu(strategy, triangle_count, static_cast<float>(triangle_count));
    restir_stream_initial_strategy_gpu(result, strategy, total, rng);

    restir_clear_initial_gpu(strategy);
    for (int i = 0; i < directional_count; ++i) {
        GpuRestirLightSample sample;
        if (!restir_sample_directional_gpu(scene, rng, sample)) continue;
        Vec3 direction; float distance;
        const Vec3 c = restir_evaluate_directional_sample_gpu(
            scene, hit, material, wo, sample, settings, direction, distance);
        restir_add_initial_gpu(strategy, sample, restir_luminance_gpu(c),
            1.0f / static_cast<float>(scene.directional_light_count), rng);
    }
    restir_finalize_initial_strategy_gpu(strategy, directional_count, static_cast<float>(directional_count));
    restir_stream_initial_strategy_gpu(result, strategy, total, rng);

    restir_clear_initial_gpu(strategy);
    for (int i = 0; i < point_count; ++i) {
        GpuRestirLightSample sample;
        if (!restir_sample_point_gpu(scene, rng, sample)) continue;
        Vec3 direction; float distance;
        const Vec3 c = restir_evaluate_point_sample_gpu(
            scene, hit, material, wo, sample, settings, direction, distance);
        restir_add_initial_gpu(strategy, sample, restir_luminance_gpu(c),
            1.0f / static_cast<float>(scene.point_light_count), rng);
    }
    restir_finalize_initial_strategy_gpu(strategy, point_count, static_cast<float>(point_count));
    restir_stream_initial_strategy_gpu(result, strategy, total, rng);

    restir_clear_initial_gpu(strategy);
    for (int i = 0; i < environment_count; ++i) {
        GpuRestirLightSample sample; float pdf = 0.0f;
        if (!restir_sample_environment_gpu(scene, rng, sample, pdf)) continue;
        Vec3 direction; float distance;
        const Vec3 c = restir_evaluate_environment_sample_gpu(
            scene, hit, material, wo, settings, sample, direction, distance);
        restir_add_initial_gpu(strategy, sample, restir_luminance_gpu(c), pdf, rng);
    }
    restir_finalize_initial_strategy_gpu(strategy, environment_count, static_cast<float>(environment_count));
    restir_stream_initial_strategy_gpu(result, strategy, total, rng);
    restir_finalize_gpu(result, 1.0f, 1.0f);
    result.M = 1;
    return result;
}

__global__ void restir_gi_generate_secondary_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuHit* hits, const int* bsdf_indices, int* next_indices,
    GpuWavefrontQueueCounters* queue_counters, Vec3* samples,
    GpuRestirGiInitialSample* initial_samples, GpuRestirSurface* current_surfaces) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = queue_counters->num_queued[GpuWavefrontQueueBsdf];
    if (queue_index >= count) return;
    const int path_index = bsdf_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;
    const GpuScene& scene = *scene_ptr;
    const GpuHit hit = hits[path_index];
    const GpuMaterial material = scene.materials[hit.material];
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal, wo,
        hit.uv, hit.front_face, path.rng);
    if (!(sample.pdf > 0.0f) || !finite_vec_gpu(sample.weight)) {
        finish_wavefront_path(path, samples);
        return;
    }
    if (sample.delta) {
        path.previous_position = hit.position;
        path.previous_bsdf_pdf = sample.pdf;
        wavefront_set_previous_delta(path, true);
        if (material_transmission_gpu(scene, material, hit.uv) > 0.5f && wavefront_transmission_bounces(path) < 12)
            wavefront_increment_transmission_bounce(path);
        const float side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
        path.ray = {add(hit.position, mul(hit.normal, 0.001f * side)), sample.direction};
        path.throughput = mul(path.throughput, sample.weight);
        wavefront_increment_bounce(path);
        wavefront_append_queue(next_indices, &queue_counters->num_queued[GpuWavefrontQueueNextRay], path_index);
        return;
    }

    const int pixel = path.pixel;
    GpuRestirGiInitialSample initial;
    const float side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
    initial.ray = {add(hit.position, mul(hit.normal, 0.001f * side)), sample.direction};
    initial.sample_weight = sample.weight;
    initial.sample_pdf = sample.pdf;
    initial.path_index = path_index;
    initial.valid = 1;
    initial.primary_surface.position = hit.position;
    initial.primary_surface.normal = hit.normal;
    initial.primary_surface.geo_normal = restir_geometric_normal_gpu(scene, hit);
    initial.primary_surface.wo = wo;
    initial.primary_surface.uv = hit.uv;
    initial.primary_surface.material = hit.material;
    initial.primary_surface.object_id = primary_object_id_gpu(hit);
    const Vec3 forward = dnormalize(sub(scene.camera.target, scene.camera.position));
    initial.primary_surface.depth = fmaxf(0.0f, ddot(sub(hit.position, scene.camera.position), forward));
    initial.primary_surface.valid = 1;
    initial_samples[pixel] = initial;
    current_surfaces[pixel] = initial.primary_surface;
}

template <bool AlphaVisibility, bool TwoLevel, GpuTraversalLayout Layout>
__global__ void restir_gi_trace_secondary_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const int* bsdf_indices, const GpuWavefrontQueueCounters* queue_counters,
    const GpuRestirGiInitialSample* initial_samples, GpuCompactHit* secondary_hits, int* results) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = queue_counters->num_queued[GpuWavefrontQueueBsdf];
    if (queue_index >= count) return;
    const int path_index = bsdf_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    const int pixel = path.pixel;
    const GpuRestirGiInitialSample initial = initial_samples[pixel];
    if (!initial.valid || initial.path_index != path_index) return;
    const GpuScene& scene = *scene_ptr;
    Ray ray = initial.ray;
    if constexpr (!AlphaVisibility) {
        GpuCompactHit hit{};
        if (!intersect_compact_gpu<TwoLevel, Layout>(scene, ray, hit)) {
            results[pixel] = 2;
            return;
        }
        hit.material = hit.triangle >= 0 ? traversal_material_index(hit.material) : hit.material;
        secondary_hits[pixel] = hit;
        results[pixel] = 1;
    } else {
        for (int step = 0; step < 8; ++step) {
            GpuCompactHit hit{};
            if (!intersect_compact_gpu<TwoLevel, Layout>(scene, ray, hit)) {
                results[pixel] = 2;
                return;
            }
            const int material_index = hit.triangle >= 0 ? traversal_material_index(hit.material) : hit.material;
            const GpuMaterial material = scene.materials[material_index];
            const Vec2 uv = compact_hit_uv_gpu(scene, ray, hit);
            if (!material_visible_gpu(scene, material, uv, path.rng)) {
                const Vec3 position = compact_hit_position_gpu(ray, hit);
                ray = {add(position, mul(ray.direction, 0.002f)), ray.direction};
                continue;
            }
            hit.material = material_index;
            secondary_hits[pixel] = hit;
            results[pixel] = 1;
            return;
        }
        results[pixel] = 0;
    }
}

__global__ void restir_gi_setup_secondary_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const int* bsdf_indices, const GpuWavefrontQueueCounters* queue_counters,
    const GpuCompactHit* compact_hits, GpuHit* secondary_hits, int* results,
    const GpuRestirGiInitialSample* initial_samples, Vec3* samples) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = queue_counters->num_queued[GpuWavefrontQueueBsdf];
    if (queue_index >= count) return;
    const int path_index = bsdf_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    const int pixel = path.pixel;
    const GpuRestirGiInitialSample initial = initial_samples[pixel];
    if (!initial.valid || initial.path_index != path_index) return;
    const GpuScene& scene = *scene_ptr;
    if (results[pixel] == 2) {
        if (!restir_direct_enabled_gpu(settings)) {
            path.radiance = add(path.radiance, clamp_sample_radiance_gpu(
                mul(initial.sample_weight, environment_radiance_gpu(scene, initial.ray.direction, settings)), 8.0f));
        }
        finish_wavefront_path(path, samples);
        return;
    }
    GpuHit hit;
    if (results[pixel] != 1 || !fill_compact_hit_gpu(scene, initial.ray, compact_hits[pixel], hit)) {
        finish_wavefront_path(path, samples);
        return;
    }
    GpuMaterial material = scene.materials[hit.material];
    hit.normal = apply_normal_map_gpu(scene, material, hit, initial.ray.direction);
    const Vec3 material_emission = material_emission_gpu(scene, material, hit.uv, settings);
    hit.emission = add(hit.emission, material_emission);
    if (has_light_emission_gpu(hit.emission)) {
        results[pixel] = 4;
        if (!restir_direct_enabled_gpu(settings) && settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling) {
            const GpuTriangle light = hit.triangle >= 0 ? scene.triangles[hit.triangle] : GpuTriangle{};
            const Vec3 emission = emitted_radiance_gpu(light, material, light.emission, material_emission,
                light.light_double_sided != 0, initial.ray.direction);
            const float pmf = scene.light_count > 0 ? 1.0f / static_cast<float>(scene.light_count) : 0.0f;
            const float light_pdf = hit.triangle >= 0
                ? light_pdf_solid_angle_gpu(light, material, initial.primary_surface.position, hit.position, pmf) : 0.0f;
            const float weight = mis_weight_gpu(initial.sample_pdf, light_pdf, static_cast<int>(settings.mis_heuristic));
            path.radiance = add(path.radiance, clamp_sample_radiance_gpu(
                mul(mul(initial.sample_weight, emission), weight), 8.0f));
        }
        finish_wavefront_path(path, samples);
        return;
    }
    results[pixel] = 3;
    secondary_hits[pixel] = hit;
}

template <bool HasLocalLights, bool HasEnvironment>
__global__ void restir_gi_sample_secondary_direct_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const int* bsdf_indices, const GpuWavefrontQueueCounters* queue_counters,
    const GpuHit* secondary_hits, const int* results,
    const GpuRestirGiInitialSample* initial_samples, GpuRestirReservoir* secondary_direct,
    GpuRestirVisibilityRay* visibility_rays, int* visibility_results,
    int* visibility_indices, int* visibility_count) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = queue_counters->num_queued[GpuWavefrontQueueBsdf];
    if (queue_index >= count) return;
    const int path_index = bsdf_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    const int pixel = path.pixel;
    const GpuRestirGiInitialSample initial = initial_samples[pixel];
    if (!initial.valid || initial.path_index != path_index || results[pixel] != 3) return;
    const GpuScene& scene = *scene_ptr;
    const GpuHit hit = secondary_hits[pixel];
    GpuMaterial material = restir_gi_roughened_material_gpu(scene, scene.materials[hit.material], hit.uv,
        settings.cuda_restir_gi_secondary_roughness);
    visibility_rays[pixel] = {};
    visibility_results[pixel] = -1;
    const Vec3 wo = mul(initial.ray.direction, -1.0f);
    GpuRestirReservoir direct = restir_gi_sample_secondary_direct_gpu<HasLocalLights, HasEnvironment>(
        scene, settings, hit, material, wo, path.rng);
    secondary_direct[pixel] = direct;
    if (!direct.valid || !(direct.weight_sum > 0.0f)) return;
    Vec3 direction; float distance = 0.0f;
    const Vec3 contribution = restir_evaluate_sample_gpu(scene, hit, material, wo,
        settings, direct.sample, direction, distance);
    if (!(restir_luminance_gpu(contribution) > 0.0f)) return;
    const float side = ddot(hit.normal, direction) >= 0.0f ? 1.0f : -1.0f;
    GpuRestirVisibilityRay visibility;
    visibility.ray = {add(hit.position, mul(hit.normal, 0.002f * side)), direction};
    visibility.remaining = isfinite(distance) ? distance - 0.01f : kInfinity;
    visibility.sample_type = static_cast<int>(direct.sample.type);
    visibility.sample_index = direct.sample.index;
    visibility.path_index = path_index;
    visibility.valid = 1;
    visibility_rays[pixel] = visibility;
    visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
}

__global__ void restir_gi_build_initial_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const int* bsdf_indices, const GpuWavefrontQueueCounters* queue_counters,
    const GpuRestirGiInitialSample* initial_samples, const GpuHit* secondary_hits,
    const int* secondary_results, const GpuRestirReservoir* secondary_direct, const int* visibility_results,
    GpuPackedRestirGiReservoir* initial_reservoirs) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = queue_counters->num_queued[GpuWavefrontQueueBsdf];
    if (queue_index >= count) return;
    const int path_index = bsdf_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    const int pixel = path.pixel;
    const GpuRestirGiInitialSample initial = initial_samples[pixel];
    GpuRestirGiReservoir reservoir = {};
    if (initial.valid && initial.path_index == path_index) {
        const GpuScene& scene = *scene_ptr;
        reservoir.weight_sum = initial.sample_pdf > 0.0f ? 1.0f / initial.sample_pdf : 0.0f;
        reservoir.M = 1;
        if (secondary_results[pixel] == 3) {
            const GpuHit hit = secondary_hits[pixel];
            reservoir.position = hit.position;
            reservoir.normal = hit.normal;
            reservoir.primitive_type = hit.sphere >= 0 ? 1 : 0;
            reservoir.primitive_index = hit.sphere >= 0 ? hit.sphere : hit.triangle;
        }
        if (secondary_results[pixel] == 3 && visibility_results[pixel] > 0) {
            const GpuHit hit = secondary_hits[pixel];
            GpuMaterial material = restir_gi_roughened_material_gpu(scene,
                scene.materials[hit.material], hit.uv, settings.cuda_restir_gi_secondary_roughness);
            const GpuRestirReservoir direct = secondary_direct[pixel];
            Vec3 direction; float distance = 0.0f;
            const Vec3 contribution = restir_evaluate_sample_gpu(scene, hit, material,
                mul(initial.ray.direction, -1.0f), settings, direct.sample, direction, distance);
            reservoir.radiance = mul(contribution, direct.weight_sum);
        }
    }
    initial_reservoirs[pixel] = restir_gi_pack(reservoir);
}

__global__ void restir_gi_temporal_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    const GpuRestirSurface* current_surfaces, const GpuRestirSurface* history_surfaces,
    const GpuPackedRestirGiReservoir* initial, const GpuPackedRestirGiReservoir* history,
    GpuPackedRestirGiReservoir* output, GpuRestirGiTemporalState* states,
    Camera history_camera, bool history_valid,
    uint32_t sequence_index, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirSurface current_surface = current_surfaces[pixel];
    if (!current_surface.valid) { output[pixel] = {}; states[pixel] = {}; return; }
    const GpuScene& scene = *scene_ptr;
    GpuRestirGiReservoir current = restir_gi_unpack(initial[pixel]);
    GpuRestirGiReservoir previous;
    GpuRestirSurface previous_surface;
    bool found = false;
    int previous_pixel = -1;
    if (history_valid && settings.cuda_restir_gi_resampling != RestirGiResamplingMode::None) {
        float projected_x = 0.0f, projected_y = 0.0f, expected_depth = 0.0f;
        if (restir_project_to_camera_gpu(history_camera, current_surface.position,
                settings.width, settings.height, projected_x, projected_y, expected_depth)) {
            const int px = static_cast<int>(roundf(projected_x));
            const int py = static_cast<int>(roundf(projected_y));
            uint32_t search_rng = restir_rng_seed_gpu(static_cast<uint32_t>(pixel) ^ sequence_index, 0x5bd1e995u);
            const int temporal_start = static_cast<int>(rng_float(search_rng) * 8.0f);
            float best_score = kInfinity;
            int best_pixel = -1;
            GpuRestirSurface best_surface;
            GpuRestirGiReservoir best_reservoir;
            for (int i = 0; i < 5; ++i) {
                const int2 offset = i == 0 ? make_int2(0, 0) : restir_gi_temporal_offset_gpu(temporal_start + i);
                const int x = px + offset.x;
                const int y = py + offset.y;
                if (x < 0 || y < 0 || x >= settings.width || y >= settings.height) continue;
                const int candidate_pixel = y * settings.width + x;
                const GpuRestirSurface candidate_surface = history_surfaces[candidate_pixel];
                if (!restir_gi_surfaces_compatible_gpu(current_surface, candidate_surface, expected_depth)) continue;
                GpuRestirGiReservoir candidate = restir_gi_unpack(history[candidate_pixel]);
                if (!restir_gi_valid(candidate) || candidate.age >= kRestirGiMaxAge) continue;
                const float dx = static_cast<float>(x) - projected_x;
                const float dy = static_cast<float>(y) - projected_y;
                const float depth_error = fabsf(candidate_surface.depth - expected_depth) /
                    fmaxf(0.01f, expected_depth);
                const float score = dx * dx + dy * dy + 4.0f * depth_error * depth_error;
                if (score < best_score) {
                    best_score = score;
                    best_pixel = candidate_pixel;
                    best_surface = candidate_surface;
                    best_reservoir = candidate;
                }
            }
            if (best_pixel >= 0) {
                previous = best_reservoir;
                previous.M = previous.M < kRestirGiMaxHistoryLength ? previous.M : kRestirGiMaxHistoryLength;
                ++previous.age;
                previous_surface = best_surface;
                previous_pixel = best_pixel;
                found = true;
            }
            if (!found) {
                const int x = pixel % settings.width;
                const int y = pixel / settings.width;
                const GpuRestirSurface candidate_surface = history_surfaces[y * settings.width + x];
                if (restir_gi_surfaces_compatible_gpu(current_surface, candidate_surface,
                        current_surface.depth)) {
                    GpuRestirGiReservoir candidate = restir_gi_unpack(history[y * settings.width + x]);
                    if (restir_gi_valid(candidate) && candidate.age < kRestirGiMaxAge) {
                        previous = candidate;
                        previous.M = previous.M < kRestirGiMaxHistoryLength ? previous.M : kRestirGiMaxHistoryLength;
                        ++previous.age;
                        previous_surface = candidate_surface;
                        previous_pixel = y * settings.width + x;
                        found = true;
                    }
                }
            }
        }
    }
    GpuRestirGiReservoir result = {};
    uint32_t rng = restir_rng_seed_gpu(static_cast<uint32_t>(pixel) ^ sequence_index, 0x91e10da5u);
    bool selected_previous = false;
    if (restir_gi_valid(current)) {
        const float target = restir_gi_target_gpu(scene, current_surface, current);
        restir_gi_combine_gpu(result, current, target, 1.0f, 0.5f);
    }
    if (found) {
        const float jacobian = restir_gi_jacobian_gpu(current_surface.position,
            previous_surface.position, previous.position, restir_gi_sample_geo_normal_gpu(scene, previous));
        if (jacobian > 0.0f) {
            const float target = restir_gi_target_gpu(scene, current_surface, previous);
            if (restir_gi_combine_gpu(result, previous, target, jacobian, rng_float(rng))) {
                selected_previous = true;
            }
        } else {
            found = false;
            previous_pixel = -1;
        }
    }
    GpuRestirGiTemporalState state;
    state.previous_pixel = found ? previous_pixel : -1;
    state.previous_m = found ? previous.M : 0;
    state.selected_previous = selected_previous ? 1 : 0;
    state.correction_rng = rng;
    states[pixel] = state;
    output[pixel] = restir_gi_pack(result);
}

__global__ void restir_gi_temporal_finalize_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    const GpuRestirSurface* current_surfaces, const GpuRestirSurface* history_surfaces,
    const GpuPackedRestirGiReservoir* initial, const GpuRestirGiTemporalState* states,
    GpuPackedRestirGiReservoir* temporal, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirSurface current_surface = current_surfaces[pixel];
    GpuRestirGiReservoir result = restir_gi_unpack(temporal[pixel]);
    if (!current_surface.valid || !restir_gi_valid(result)) return;
    const GpuScene& scene = *scene_ptr;
    const GpuRestirGiReservoir canonical = restir_gi_unpack(initial[pixel]);
    const GpuRestirGiTemporalState state = states[pixel];
    const float selected_target = restir_gi_target_gpu(scene, current_surface, result);
    float pi = selected_target;
    float pi_sum = selected_target * static_cast<float>(canonical.M);
    if (state.previous_pixel >= 0 && state.previous_m > 0) {
        const GpuRestirSurface previous_surface = history_surfaces[state.previous_pixel];
        float previous_p = restir_gi_target_gpu(scene, previous_surface, result);
        uint32_t rng = state.correction_rng;
        if (settings.cuda_restir_gi_bias_correction == RestirBiasCorrection::RayTraced && previous_p > 0.0f &&
            !restir_gi_visible_gpu(scene, previous_surface.position, previous_surface.normal, result.position, rng))
            previous_p = 0.0f;
        if (state.selected_previous != 0) pi = previous_p;
        pi_sum += previous_p * static_cast<float>(state.previous_m);
    }
    restir_gi_finalize_gpu(result, pi, selected_target * pi_sum);
    temporal[pixel] = restir_gi_pack(result);
}

__global__ void restir_gi_boiling_filter_kernel(const GpuPackedRestirGiReservoir* input,
    GpuPackedRestirGiReservoir* output, int width, int height) {
    __shared__ float weights[256];
    __shared__ int nonzero[256];
    const int local = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const bool inside = x < width && y < height;
    const int pixel = y * width + x;
    const GpuRestirGiReservoir reservoir = inside ? restir_gi_unpack(input[pixel]) : GpuRestirGiReservoir{};
    const float value = restir_gi_valid(reservoir)
        ? restir_luminance_gpu(reservoir.radiance) * reservoir.weight_sum : 0.0f;
    weights[local] = value;
    nonzero[local] = value > 0.0f ? 1 : 0;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (local < stride) {
            weights[local] += weights[local + stride];
            nonzero[local] += nonzero[local + stride];
        }
        __syncthreads();
    }
    if (!inside) return;
    const float average = nonzero[0] > 0 ? weights[0] / static_cast<float>(nonzero[0]) : 0.0f;
    const float multiplier = 10.0f / kRestirGiBoilingStrength - 9.0f;
    output[pixel] = value > average * multiplier ? GpuPackedRestirGiReservoir{} : input[pixel];
}

__device__ int restir_gi_reflect_index(int value, int extent) {
    if (value < 0) value = -value;
    if (value >= extent) value = 2 * extent - value - 1;
    return iclamp_gpu(value, 0, extent - 1);
}

__global__ void restir_gi_spatial_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    const GpuRestirSurface* surfaces, const GpuPackedRestirGiReservoir* input,
    GpuPackedRestirGiReservoir* output, GpuRestirGiSpatialState* states,
    uint32_t sequence_index, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirSurface surface = surfaces[pixel];
    if (!surface.valid) { output[pixel] = {}; states[pixel] = {}; return; }
    const GpuScene& scene = *scene_ptr;
    const int x = pixel % settings.width;
    const int y = pixel / settings.width;
    uint32_t rng = restir_rng_seed_gpu(static_cast<uint32_t>(pixel) ^ sequence_index, 0xd1b54a35u);
    GpuRestirGiReservoir result = {};
    const GpuRestirGiReservoir canonical = restir_gi_unpack(input[pixel]);
    const float canonical_target = restir_gi_target_gpu(scene, surface, canonical);
    restir_gi_combine_gpu(result, canonical, canonical_target, 1.0f, 0.5f);
    GpuRestirGiSpatialState state;
    for (int i = 0; i < kRestirGiSpatialSamples; ++i) {
        const float angle = 2.0f * kPi * rng_float(rng);
        const float radius = kRestirGiSpatialRadius * sqrtf(rng_float(rng));
        const int nx = restir_gi_reflect_index(x + static_cast<int>(cosf(angle) * radius), settings.width);
        const int ny = restir_gi_reflect_index(y + static_cast<int>(sinf(angle) * radius), settings.height);
        const int neighbor_pixel = ny * settings.width + nx;
        const GpuRestirSurface neighbor_surface = surfaces[neighbor_pixel];
        if (!restir_gi_surfaces_compatible_gpu(surface, neighbor_surface, surface.depth)) continue;
        const GpuRestirGiReservoir neighbor = restir_gi_unpack(input[neighbor_pixel]);
        if (!restir_gi_valid(neighbor)) continue;
        const float jacobian = restir_gi_jacobian_gpu(surface.position, neighbor_surface.position,
            neighbor.position, restir_gi_sample_geo_normal_gpu(scene, neighbor));
        if (!(jacobian > 0.0f)) continue;
        const int state_index = state.neighbor_count++;
        state.neighbor_pixels[state_index] = neighbor_pixel;
        state.neighbor_m[state_index] = neighbor.M;
        const float target = restir_gi_target_gpu(scene, surface, neighbor);
        if (restir_gi_combine_gpu(result, neighbor, target, jacobian, rng_float(rng))) {
            state.selected_neighbor = state_index;
        }
    }
    state.correction_rng = rng;
    states[pixel] = state;
    output[pixel] = restir_gi_pack(result);
}

__global__ void restir_gi_spatial_finalize_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    const GpuRestirSurface* surfaces, const GpuPackedRestirGiReservoir* input,
    const GpuRestirGiSpatialState* states, GpuPackedRestirGiReservoir* output, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirSurface surface = surfaces[pixel];
    GpuRestirGiReservoir result = restir_gi_unpack(output[pixel]);
    if (!surface.valid || !restir_gi_valid(result)) return;
    const GpuScene& scene = *scene_ptr;
    const GpuRestirGiReservoir canonical = restir_gi_unpack(input[pixel]);
    const GpuRestirGiSpatialState state = states[pixel];
    const float selected_target = restir_gi_target_gpu(scene, surface, result);
    float pi = selected_target;
    float pi_sum = selected_target * static_cast<float>(canonical.M);
    uint32_t rng = state.correction_rng;
    for (int i = 0; i < state.neighbor_count; ++i) {
        const int neighbor_pixel = state.neighbor_pixels[i];
        if (neighbor_pixel < 0 || state.neighbor_m[i] <= 0) continue;
        const GpuRestirSurface neighbor_surface = surfaces[neighbor_pixel];
        float neighbor_p = restir_gi_target_gpu(scene, neighbor_surface, result);
        if (settings.cuda_restir_gi_bias_correction == RestirBiasCorrection::RayTraced && neighbor_p > 0.0f &&
            !restir_gi_visible_gpu(scene, neighbor_surface.position, neighbor_surface.normal, result.position, rng))
            neighbor_p = 0.0f;
        if (state.selected_neighbor == i) pi = neighbor_p;
        pi_sum += neighbor_p * static_cast<float>(state.neighbor_m[i]);
    }
    restir_gi_finalize_gpu(result, pi, selected_target * pi_sum);
    output[pixel] = restir_gi_pack(result);
}

__global__ void restir_gi_generate_final_visibility_kernel(const GpuScene* scene_ptr,
    const GpuRestirSurface* surfaces, const GpuPackedRestirGiReservoir* reservoirs, GpuRestirVisibilityRay* rays,
    int* results, const GpuRestirGiInitialSample* initial_samples,
    int* visibility_indices, int* visibility_count, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    rays[pixel] = {};
    results[pixel] = -1;
    const GpuRestirSurface surface = surfaces[pixel];
    const GpuRestirGiReservoir reservoir = restir_gi_unpack(reservoirs[pixel]);
    if (!surface.valid || !restir_gi_valid(reservoir)) return;
    if (!(reservoir.weight_sum > 0.0f) || !(restir_luminance_gpu(reservoir.radiance) > 0.0f)) return;
    const GpuScene& scene = *scene_ptr;
    const Vec3 delta = sub(reservoir.position, surface.position);
    const float distance = sqrtf(ddot(delta, delta));
    if (!(distance > 0.003f)) return;
    const Vec3 direction = divv(delta, distance);
    if (!restir_gi_sample_faces_receiver_gpu(scene, reservoir, surface.position)) return;
    GpuRestirVisibilityRay ray;
    ray.ray = {add(surface.position, mul(direction, 0.002f)), direction};
    ray.remaining = distance;
    ray.sample_type = reservoir.primitive_type != 0
        ? kRestirVisibilityGiSphere : kRestirVisibilityGiTriangle;
    ray.sample_index = reservoir.primitive_index;
    ray.path_index = initial_samples[pixel].path_index;
    ray.valid = 1;
    rays[pixel] = ray;
    visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
}

__device__ Vec3 restir_gi_contribution_gpu(const GpuScene& scene, const GpuRestirSurface& surface,
    const GpuRestirGiReservoir& reservoir, GpuMaterial material) {
    const Vec3 delta = sub(reservoir.position, surface.position);
    const float distance2 = ddot(delta, delta);
    if (!(distance2 > 1.0e-10f)) return {};
    const Vec3 wi = mul(delta, rsqrtf(distance2));
    if (!restir_gi_receiver_accepts_direction_gpu(surface, material, wi) ||
        !restir_gi_sample_faces_receiver_gpu(scene, reservoir, surface.position)) return {};
    const float cosine = (material.double_sided || material.brdf_model == static_cast<int>(BrdfModel::DiffuseTransmission))
        ? fabsf(ddot(surface.normal, wi)) : fmaxf(0.0f, ddot(surface.normal, wi));
    return mul(mul(evaluate_brdf_gpu(scene, material, surface.normal, surface.wo, wi, surface.uv),
        reservoir.radiance), cosine * reservoir.weight_sum);
}

__device__ Vec3 restir_gi_brdf_term_gpu(const GpuScene& scene, const GpuRestirSurface& surface,
    const GpuRestirGiReservoir& reservoir, const GpuMaterial& material) {
    const Vec3 delta = sub(reservoir.position, surface.position);
    const float distance2 = ddot(delta, delta);
    if (!(distance2 > 1.0e-10f)) return {};
    const Vec3 wi = mul(delta, rsqrtf(distance2));
    if (!restir_gi_receiver_accepts_direction_gpu(surface, material, wi) ||
        !restir_gi_sample_faces_receiver_gpu(scene, reservoir, surface.position)) return {};
    const float cosine = (material.double_sided || material.brdf_model == static_cast<int>(BrdfModel::DiffuseTransmission))
        ? fabsf(ddot(surface.normal, wi)) : fmaxf(0.0f, ddot(surface.normal, wi));
    return mul(evaluate_brdf_gpu(scene, material, surface.normal, surface.wo, wi, surface.uv), cosine);
}

__device__ float restir_gi_final_mis_weight_gpu(Vec3 rough_brdf, Vec3 true_brdf) {
    rough_brdf = {
        dclamp(rough_brdf.x, 1.0e-4f, 1.0e4f),
        dclamp(rough_brdf.y, 1.0e-4f, 1.0e4f),
        dclamp(rough_brdf.z, 1.0e-4f, 1.0e4f),
    };
    true_brdf = {
        dclamp(true_brdf.x, 0.0f, 1.0e4f),
        dclamp(true_brdf.y, 0.0f, 1.0e4f),
        dclamp(true_brdf.z, 0.0f, 1.0e4f),
    };
    const float weight = dclamp(restir_luminance_gpu(true_brdf) /
        fmaxf(1.0e-8f, restir_luminance_gpu(add(true_brdf, rough_brdf))), 0.0f, 1.0f);
    return weight * weight * weight;
}

__global__ void restir_gi_resolve_final_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuRestirGiInitialSample* initial_samples,
    const GpuRestirSurface* surfaces, const GpuPackedRestirGiReservoir* initial_reservoirs,
    const GpuPackedRestirGiReservoir* final_reservoirs, const int* visibility_results,
    Vec3* samples, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirGiInitialSample initial_sample = initial_samples[pixel];
    if (!initial_sample.valid || initial_sample.path_index < 0) return;
    GpuWavefrontPathRef path = wavefront_path_ref(paths, initial_sample.path_index);
    const GpuRestirSurface surface = surfaces[pixel];
    const GpuScene& scene = *scene_ptr;
    GpuMaterial material = scene.materials[surface.material];
    const GpuRestirGiReservoir final_reservoir = restir_gi_unpack(final_reservoirs[pixel]);
    Vec3 indirect{};
    if (restir_gi_valid(final_reservoir)) {
        const Vec3 final_true = visibility_results[pixel] > 0
            ? restir_gi_contribution_gpu(scene, surface, final_reservoir, material) : Vec3{};
        indirect = final_true;
        if (settings.cuda_restir_gi_final_mis) {
            const GpuRestirGiReservoir initial_reservoir = restir_gi_unpack(initial_reservoirs[pixel]);
            if (restir_gi_valid(initial_reservoir)) {
                const GpuMaterial rough_material = restir_gi_roughened_material_gpu(
                    scene, material, surface.uv, kRestirGiFinalMisRoughness);
                const Vec3 initial_true = restir_gi_contribution_gpu(scene, surface, initial_reservoir, material);
                const float final_weight = 1.0f - restir_gi_final_mis_weight_gpu(
                    restir_gi_brdf_term_gpu(scene, surface, final_reservoir, rough_material),
                    restir_gi_brdf_term_gpu(scene, surface, final_reservoir, material));
                const float initial_weight = restir_gi_final_mis_weight_gpu(
                    restir_gi_brdf_term_gpu(scene, surface, initial_reservoir, rough_material),
                    restir_gi_brdf_term_gpu(scene, surface, initial_reservoir, material));
                indirect = add(mul(final_true, final_weight), mul(initial_true, initial_weight));
            }
        }
    }
    path.radiance = add(path.radiance, clamp_sample_radiance_gpu(mul(path.throughput, indirect), 8.0f));
    finish_wavefront_path(path, samples);
}

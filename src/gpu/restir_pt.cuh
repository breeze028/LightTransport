// CUDA-native ReSTIR PT for the wavefront renderer. The implementation mirrors
// the GRIS/RTXDI data flow while using the renderer's own BSDF and light APIs.

struct GpuRestirPtReservoir {
    Vec3 position = {};
    Vec3 normal = {};
    Vec3 radiance = {};
    Vec3 target = {};
    float weight_sum = 0.0f;
    float partial_jacobian = 0.0f;
    float rc_wi_pdf = 0.0f;
    int M = 0;
    int age = 0;
    int rc_length = 0;
    int path_length = 0;
    int random_index = 0;
    int rr_count = 0;
    uint32_t random_seed = 0u;
    uint32_t flags = 0u;
};

// Four 16-byte blocks, matching the persistent footprint used by RTXDI PT.
struct GpuPackedRestirPtReservoir {
    Vec3 position = {};
    float weight_sum = 0.0f;
    uint32_t packed_normal = 0u;
    uint32_t packed_m_age = 0u;
    float partial_jacobian = 0.0f;
    float rc_wi_pdf = 0.0f;
    Vec3 radiance = {};
    uint32_t packed_path = 0u;
    Vec3 target = {};
    uint32_t random_seed = 0u;
};

static_assert(sizeof(GpuPackedRestirPtReservoir) == 64,
    "packed ReSTIR PT reservoir must be 64 bytes");

enum GpuRestirPtFlags : uint32_t {
    GpuRestirPtEnvironment = 1u << 0u,
    GpuRestirPtDeltaPrefix = 1u << 1u,
    GpuRestirPtNeeLight = 1u << 2u,
};

struct GpuRestirPtPathState {
    Ray ray = {};
    GpuHit hit = {};
    GpuRestirSurface primary_surface = {};
    GpuRestirPtReservoir selected = {};
    Vec3 throughput = {1.0f, 1.0f, 1.0f};
    Vec3 rc_throughput = {1.0f, 1.0f, 1.0f};
    Vec3 previous_position = {};
    float previous_pdf = 0.0f;
    float candidate_weight_sum = 0.0f;
    float footprint_threshold = 0.02f;
    float pdf_threshold = 0.1f;
    uint32_t replay_rng = 0u;
    uint32_t resampling_rng = 0u;
    uint32_t nee_rng = 0u;
    int path_index = -1;
    int depth = 1;
    int active = 0;
    int trace_result = 0;
    int previous_delta = 0;
    int previous_rough = 0;
    int previous_far = 0;
    int rc_found = 0;
    int rr_count = 0;
};

struct GpuRestirPtQueueCounters {
    int active_count = 0;
    int next_count = 0;
};

struct GpuRestirPtTemporalState {
    int source_pixel = -1;
    int source_m = 0;
    int selected_source = 0;
    int valid = 0;
    int replayed = 0;
    float jacobian = 0.0f;
};

struct GpuRestirPtSpatialState {
    int source_pixel = -1;
    int source_m = 0;
    int selected_source = 0;
    int valid = 0;
    int replayed = 0;
    float jacobian = 0.0f;
};

static constexpr int kRestirPtMaxHistoryLength = 8;
static constexpr int kRestirPtMaxAge = 30;
static constexpr float kRestirPtDepthThreshold = 0.10f;
static constexpr float kRestirPtNormalThreshold = 0.60f;
static constexpr float kRestirPtThroughputCutoff = 0.05f;
static constexpr float kRestirPtRrProbability = 0.50f;
static constexpr int kRestirPtExtraDeltaBudget = 4;
static constexpr float kRestirPtBoilingStrength = 0.20f;
static constexpr float kRestirPtHistoryReductionStrength = 0.80f;
static constexpr int kRestirPtPairingMapCount = 3;
static constexpr float kRestirPtPairingSigma = 16.0f;

__device__ bool restir_pt_valid_gpu(const GpuRestirPtReservoir& reservoir) {
    return reservoir.M > 0 && isfinite(reservoir.weight_sum) &&
        finite_vec_gpu(reservoir.target) &&
        ((reservoir.flags & GpuRestirPtNeeLight) != 0u || finite_vec_gpu(reservoir.radiance));
}

__device__ int restir_pt_reduced_history_length_gpu(uint32_t duplicate_count) {
    const float impoverishment = dclamp(
        static_cast<float>(duplicate_count) / 288.0f, 0.0f, 1.0f);
    const float power_factor = 0.1f * exp2f(
        6.0f * (1.0f - kRestirPtHistoryReductionStrength) - 3.0f);
    const float reduction = powf(impoverishment, power_factor);
    return iclamp_gpu(static_cast<int>(
        static_cast<float>(kRestirPtMaxHistoryLength) * (1.0f - reduction) + reduction),
        1, kRestirPtMaxHistoryLength);
}

__device__ Vec3 restir_pt_pack_light_sample_gpu(const GpuRestirLightSample& sample) {
    const uint32_t identity =
        ((static_cast<uint32_t>(sample.type) & 0x3u) << 30u) |
        (static_cast<uint32_t>(sample.index + 1) & 0x3fffffffu);
    const uint32_t u = static_cast<uint32_t>(dclamp(sample.uv.x, 0.0f, 1.0f) * 65535.0f + 0.5f);
    const uint32_t v = static_cast<uint32_t>(dclamp(sample.uv.y, 0.0f, 1.0f) * 65535.0f + 0.5f);
    return {__uint_as_float(identity), __uint_as_float(u | (v << 16u)),
        __uint_as_float(static_cast<uint32_t>(sample.sampler_index + 1))};
}

__device__ GpuRestirLightSample restir_pt_unpack_light_sample_gpu(
    const GpuRestirPtReservoir& reservoir) {
    GpuRestirLightSample sample;
    const uint32_t identity = __float_as_uint(reservoir.radiance.x);
    const uint32_t packed_uv = __float_as_uint(reservoir.radiance.y);
    sample.type = static_cast<GpuRestirLightType>((identity >> 30u) & 0x3u);
    sample.index = static_cast<int>(identity & 0x3fffffffu) - 1;
    sample.position_or_direction = reservoir.position;
    sample.uv = {static_cast<float>(packed_uv & 0xffffu) / 65535.0f,
        static_cast<float>((packed_uv >> 16u) & 0xffffu) / 65535.0f};
    sample.sampler_index = static_cast<int>(__float_as_uint(reservoir.radiance.z)) - 1;
    return sample;
}

__device__ GpuPackedRestirPtReservoir restir_pt_pack_gpu(const GpuRestirPtReservoir& reservoir) {
    GpuPackedRestirPtReservoir packed;
    if (reservoir.M <= 0) return packed;
    packed.position = reservoir.position;
    packed.weight_sum = reservoir.weight_sum;
    packed.packed_normal = restir_gi_pack_snorm2(reservoir.normal);
    packed.packed_m_age = static_cast<uint32_t>(iclamp_gpu(reservoir.M, 0, 0xffff)) |
        (static_cast<uint32_t>(iclamp_gpu(reservoir.age, 0, 31)) << 16u);
    packed.partial_jacobian = reservoir.partial_jacobian;
    packed.rc_wi_pdf = reservoir.rc_wi_pdf;
    packed.radiance = reservoir.radiance;
    packed.packed_path = static_cast<uint32_t>(iclamp_gpu(reservoir.rc_length, 0, 255)) |
        (static_cast<uint32_t>(iclamp_gpu(reservoir.path_length, 0, 255)) << 8u) |
        (static_cast<uint32_t>(iclamp_gpu(reservoir.random_index, 0, 15)) << 16u) |
        (static_cast<uint32_t>(iclamp_gpu(reservoir.rr_count, 0, 15)) << 20u) |
        ((reservoir.flags & 0xffu) << 24u);
    packed.target = reservoir.target;
    packed.random_seed = reservoir.random_seed;
    return packed;
}

__device__ GpuRestirPtReservoir restir_pt_unpack_gpu(const GpuPackedRestirPtReservoir& packed) {
    GpuRestirPtReservoir reservoir;
    reservoir.M = static_cast<int>(packed.packed_m_age & 0xffffu);
    if (reservoir.M <= 0) return {};
    reservoir.age = static_cast<int>((packed.packed_m_age >> 16u) & 31u);
    reservoir.position = packed.position;
    reservoir.weight_sum = packed.weight_sum;
    reservoir.normal = restir_gi_unpack_snorm2(packed.packed_normal);
    reservoir.partial_jacobian = packed.partial_jacobian;
    reservoir.rc_wi_pdf = packed.rc_wi_pdf;
    reservoir.radiance = packed.radiance;
    reservoir.rc_length = static_cast<int>(packed.packed_path & 0xffu);
    reservoir.path_length = static_cast<int>((packed.packed_path >> 8u) & 0xffu);
    reservoir.random_index = static_cast<int>((packed.packed_path >> 16u) & 0x0fu);
    reservoir.rr_count = static_cast<int>((packed.packed_path >> 20u) & 0x0fu);
    reservoir.flags = (packed.packed_path >> 24u) & 0xffu;
    reservoir.target = packed.target;
    reservoir.random_seed = packed.random_seed;
    return reservoir;
}

__global__ void restir_pt_fill_sample_ids_kernel(
    const GpuPackedRestirPtReservoir* reservoirs, uint32_t* sample_ids, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirPtReservoir reservoir = restir_pt_unpack_gpu(reservoirs[pixel]);
    sample_ids[pixel] = restir_pt_valid_gpu(reservoir) && reservoir.weight_sum > 0.0f
        ? reservoir.random_seed : 0u;
}

__global__ void restir_pt_duplication_map_kernel(const uint32_t* sample_ids,
    uint32_t* duplication_counts, int width, int height) {
    constexpr int kBlockSize = 16;
    constexpr int kSearchRadius = 8;
    constexpr int kTileSize = kBlockSize + 2 * kSearchRadius;
    __shared__ uint32_t tile[kTileSize * kTileSize];

    const int block_x = static_cast<int>(blockIdx.x) * kBlockSize - kSearchRadius;
    const int block_y = static_cast<int>(blockIdx.y) * kBlockSize - kSearchRadius;
    for (int local_y = static_cast<int>(threadIdx.y); local_y < kTileSize;
         local_y += static_cast<int>(blockDim.y)) {
        const int y = block_y + local_y;
        for (int local_x = static_cast<int>(threadIdx.x); local_x < kTileSize;
             local_x += static_cast<int>(blockDim.x)) {
            const int x = block_x + local_x;
            tile[local_y * kTileSize + local_x] =
                x >= 0 && y >= 0 && x < width && y < height
                ? sample_ids[y * width + x] : 0u;
        }
    }
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x) * kBlockSize + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * kBlockSize + static_cast<int>(threadIdx.y);
    if (x >= width || y >= height) return;
    const int center_x = static_cast<int>(threadIdx.x) + kSearchRadius;
    const int center_y = static_cast<int>(threadIdx.y) + kSearchRadius;
    const uint32_t own_id = tile[center_y * kTileSize + center_x];
    uint32_t duplicates = 0u;
    if (own_id != 0u) {
        for (int oy = -kSearchRadius; oy <= kSearchRadius; ++oy) {
            for (int ox = -kSearchRadius; ox <= kSearchRadius; ++ox) {
                if ((ox != 0 || oy != 0) &&
                    tile[(center_y + oy) * kTileSize + center_x + ox] == own_id) {
                    ++duplicates;
                }
            }
        }
    }
    duplication_counts[y * width + x] = duplicates < 288u ? duplicates : 288u;
}

__device__ Vec3 restir_pt_safe_divide_gpu(Vec3 numerator, Vec3 denominator) {
    return {
        fabsf(denominator.x) > 1.0e-6f ? numerator.x / denominator.x : 0.0f,
        fabsf(denominator.y) > 1.0e-6f ? numerator.y / denominator.y : 0.0f,
        fabsf(denominator.z) > 1.0e-6f ? numerator.z / denominator.z : 0.0f,
    };
}

__device__ float restir_pt_material_roughness_gpu(const GpuScene& scene,
    const GpuMaterial& material, Vec2 uv) {
    if (material.brdf_model == static_cast<int>(BrdfModel::Mirror) ||
        material.brdf_model == static_cast<int>(BrdfModel::Dielectric)) return 0.0f;
    return material_roughness_gpu(scene, material, uv);
}

__device__ void restir_pt_reconnection_thresholds_gpu(uint32_t seed,
    float& footprint_threshold, float& pdf_threshold) {
    uint32_t rng = restir_rng_seed_gpu(seed, 0x243f6a88u);
    footprint_threshold = 0.02f * exp2f((rng_float(rng) - 0.5f) * 0.4f);
    const float min_pdf_roughness = 0.1f * exp2f((rng_float(rng) - 0.5f) * 0.08f);
    pdf_threshold = 1.0f / fmaxf(1.0e-6f, min_pdf_roughness * min_pdf_roughness);
}

__device__ bool restir_pt_validate_reconnection_gpu(const GpuScene& scene,
    const RenderSettings& settings, const GpuRestirSurface& receiver,
    const GpuRestirPtReservoir& sample) {
    if ((sample.flags & GpuRestirPtEnvironment) != 0u) return true;
    if (!receiver.valid || receiver.material < 0 || receiver.material >= scene.material_count) return false;
    const Vec3 delta = sub(sample.position, receiver.position);
    const float distance2 = ddot(delta, delta);
    if (!(distance2 > 1.0e-10f)) return false;
    const Vec3 direction = mul(delta, rsqrtf(distance2));
    const float sample_cosine = fmaxf(0.0f, ddot(sample.normal, mul(direction, -1.0f)));
    if (!(sample_cosine > 1.0e-6f)) return false;
    const GpuMaterial material = scene.materials[receiver.material];
    const float scatter_pdf = material_pdf_gpu(scene, material, receiver.normal,
        receiver.wo, direction, receiver.uv);
    if (!(scatter_pdf > 0.0f) || !isfinite(scatter_pdf)) return false;
    if (settings.cuda_restir_pt_reconnection == RestirPtReconnectionMode::FixedThreshold) {
        return restir_pt_material_roughness_gpu(scene, material, receiver.uv) >= 0.1f;
    }
    float footprint_threshold = 0.02f;
    float pdf_threshold = 0.1f;
    restir_pt_reconnection_thresholds_gpu(sample.random_seed, footprint_threshold, pdf_threshold);
    const float footprint = distance2 / (sample_cosine * scatter_pdf);
    if (!(footprint > footprint_threshold)) return false;
    if (sample.rc_length < sample.path_length) {
        if (scatter_pdf > pdf_threshold || !(sample.rc_wi_pdf > 0.0f)) return false;
        const float receiver_cosine = fabsf(ddot(receiver.normal, direction));
        const float inverse_footprint = receiver_cosine > 1.0e-6f
            ? distance2 / (receiver_cosine * sample.rc_wi_pdf) : kInfinity;
        if (!(inverse_footprint > footprint_threshold)) return false;
    }
    return true;
}

__device__ float restir_pt_partial_inverse_jacobian_gpu(Vec3 receiver, Vec3 sample, Vec3 sample_normal) {
    const Vec3 delta = sub(receiver, sample);
    const float distance2 = ddot(delta, delta);
    if (!(distance2 > 1.0e-10f)) return 0.0f;
    const float cosine = fmaxf(0.0f, ddot(sample_normal, mul(delta, rsqrtf(distance2))));
    return cosine > 1.0e-6f ? distance2 / cosine : 0.0f;
}

__device__ float restir_pt_reconnection_jacobian_gpu(const GpuRestirSurface& receiver,
    const GpuRestirPtReservoir& sample) {
    if ((sample.flags & GpuRestirPtEnvironment) != 0u) return 1.0f;
    const Vec3 delta = sub(receiver.position, sample.position);
    const float distance2 = ddot(delta, delta);
    if (!(distance2 > 1.0e-10f) || !(sample.partial_jacobian > 0.0f)) return 0.0f;
    const float cosine = fmaxf(0.0f, ddot(sample.normal, mul(delta, rsqrtf(distance2))));
    const float jacobian = cosine * sample.partial_jacobian / distance2;
    return isfinite(jacobian) && jacobian >= 0.1f && jacobian <= 10.0f ? jacobian : 0.0f;
}

__device__ int restir_pt_light_family_count_gpu(const GpuScene& scene,
    const RenderSettings& settings) {
    return (scene.light_count > 0 ? 1 : 0) +
        (scene.directional_light_count > 0 ? 1 : 0) +
        (scene.point_light_count > 0 ? 1 : 0) +
        (settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling &&
            restir_has_environment_light_gpu(scene) ? 1 : 0);
}

__device__ float restir_pt_light_proposal_pdf_gpu(const GpuScene& scene,
    const RenderSettings& settings, const GpuRestirSurface& surface,
    const GpuRestirLightSample& sample) {
    const int family_count = restir_pt_light_family_count_gpu(scene, settings);
    if (family_count <= 0) return 0.0f;
    const float family_pdf = 1.0f / static_cast<float>(family_count);
    if (sample.type == GpuRestirLightType::Triangle) {
        return family_pdf * restir_triangle_proposal_pdf_gpu(
            scene, restir_surface_hit_gpu(surface), sample);
    }
    if (sample.type == GpuRestirLightType::Directional) {
        return scene.directional_light_count > 0
            ? family_pdf / static_cast<float>(scene.directional_light_count) : 0.0f;
    }
    if (sample.type == GpuRestirLightType::Point) {
        return scene.point_light_count > 0
            ? family_pdf / static_cast<float>(scene.point_light_count) : 0.0f;
    }
    return family_pdf * restir_environment_pdf_gpu(scene, sample.uv, sample.index);
}

__device__ float restir_pt_nee_mis_weight_gpu(const GpuScene& scene,
    const RenderSettings& settings, const GpuHit& hit, const GpuMaterial& material,
    Vec3 wo, Vec3 direction, float proposal_pdf) {
    if (settings.sampling_mode != PathSamplingMode::MultipleImportanceSampling) return 1.0f;
    const float bsdf_pdf = material_pdf_gpu(scene, material, hit.normal, wo, direction, hit.uv);
    return isfinite(bsdf_pdf) && bsdf_pdf >= 0.0f
        ? mis_weight_gpu(proposal_pdf, bsdf_pdf, static_cast<int>(settings.mis_heuristic)) : 0.0f;
}

__device__ float restir_pt_nee_mis_weight_gpu(const GpuScene& scene,
    const RenderSettings& settings, const GpuRestirSurface& surface,
    Vec3 direction, float proposal_pdf) {
    if (settings.sampling_mode != PathSamplingMode::MultipleImportanceSampling) return 1.0f;
    if (surface.material < 0 || surface.material >= scene.material_count) return 0.0f;
    const GpuMaterial material = scene.materials[surface.material];
    const float bsdf_pdf = material_pdf_gpu(scene, material, surface.normal,
        surface.wo, direction, surface.uv);
    return isfinite(bsdf_pdf) && bsdf_pdf >= 0.0f
        ? mis_weight_gpu(proposal_pdf, bsdf_pdf, static_cast<int>(settings.mis_heuristic)) : 0.0f;
}

template <bool HasLocalLights, bool HasEnvironment>
__device__ GpuRestirReservoir restir_pt_sample_nee_gpu(const GpuScene& scene,
    const RenderSettings& settings, const GpuHit& hit, const GpuMaterial& material,
    Vec3 wo, uint32_t& rng) {
    GpuRestirReservoir result;
    int family_count = 0;
    if constexpr (HasLocalLights) {
        family_count += scene.light_count > 0 ? 1 : 0;
        family_count += scene.directional_light_count > 0 ? 1 : 0;
        family_count += scene.point_light_count > 0 ? 1 : 0;
    }
    if constexpr (HasEnvironment) {
        family_count += settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling &&
            restir_has_environment_light_gpu(scene) ? 1 : 0;
    }
    if (family_count <= 0) return result;

    int choice = iclamp_gpu(static_cast<int>(rng_float(rng) * static_cast<float>(family_count)),
        0, family_count - 1);
    GpuRestirLightSample sample;
    float proposal_pdf = 0.0f;
    bool sampled = false;
    const float family_pdf = 1.0f / static_cast<float>(family_count);
    (void)family_pdf;
    if constexpr (HasLocalLights) {
        if (scene.light_count > 0) {
            if (choice == 0) {
                float discrete_pdf = 0.0f;
                sampled = restir_sample_triangle_gpu(scene, rng, sample, discrete_pdf);
                if (sampled) {
                    proposal_pdf = family_pdf * restir_triangle_proposal_pdf_gpu(scene, hit, sample);
                }
            }
            --choice;
        }
        if (!sampled && scene.directional_light_count > 0) {
            if (choice == 0) {
                sampled = restir_sample_directional_gpu(scene, rng, sample);
                proposal_pdf = sampled
                    ? family_pdf / static_cast<float>(scene.directional_light_count) : 0.0f;
            }
            --choice;
        }
        if (!sampled && scene.point_light_count > 0) {
            if (choice == 0) {
                sampled = restir_sample_point_gpu(scene, rng, sample);
                proposal_pdf = sampled
                    ? family_pdf / static_cast<float>(scene.point_light_count) : 0.0f;
            }
            --choice;
        }
    }
    if constexpr (HasEnvironment) {
        if (!sampled && settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling &&
            restir_has_environment_light_gpu(scene) && choice == 0) {
            float environment_pdf = 0.0f;
            sampled = restir_sample_environment_gpu(scene, rng, sample, environment_pdf);
            proposal_pdf = sampled ? family_pdf * environment_pdf : 0.0f;
        }
    }
    if (!sampled || !(proposal_pdf > 0.0f) || !isfinite(proposal_pdf)) return result;

    Vec3 direction;
    float distance = 0.0f;
    const Vec3 contribution = restir_evaluate_sample_gpu(
        scene, hit, material, wo, settings, sample, direction, distance);
    const float mis = restir_pt_nee_mis_weight_gpu(
        scene, settings, hit, material, wo, direction, proposal_pdf);
    const float target = restir_luminance_gpu(mul(contribution, mis));
    result.sample = sample;
    result.M = 1;
    if (!(target > 0.0f) || !isfinite(target)) return result;
    result.weight_sum = 1.0f / proposal_pdf;
    result.selected_target = target;
    result.valid = isfinite(result.weight_sum) ? 1 : 0;
    return result;
}

__device__ Vec3 restir_pt_primary_term_gpu(const GpuScene& scene,
    const GpuRestirSurface& surface, const GpuRestirPtReservoir& sample) {
    if (!surface.valid || surface.material < 0 || surface.material >= scene.material_count) return {};
    Vec3 wi;
    if ((sample.flags & GpuRestirPtEnvironment) != 0u) {
        wi = dnormalize(sample.position);
    } else {
        const Vec3 delta = sub(sample.position, surface.position);
        const float distance2 = ddot(delta, delta);
        if (!(distance2 > 1.0e-10f)) return {};
        wi = mul(delta, rsqrtf(distance2));
    }
    const GpuMaterial material = scene.materials[surface.material];
    if (!restir_gi_receiver_accepts_direction_gpu(surface, material, wi)) return {};
    const float cosine = material.double_sided ||
        material.brdf_model == static_cast<int>(BrdfModel::DiffuseTransmission)
        ? fabsf(ddot(surface.normal, wi)) : fmaxf(0.0f, ddot(surface.normal, wi));
    return mul(evaluate_brdf_gpu(scene, material, surface.normal, surface.wo, wi, surface.uv), cosine);
}

__device__ Vec3 restir_pt_shift_target_gpu(const GpuScene& scene,
    const GpuRestirSurface& target_surface, const GpuRestirSurface& source_surface,
    const GpuRestirPtReservoir& sample) {
    (void)source_surface;
    const Vec3 target_term = restir_pt_primary_term_gpu(scene, target_surface, sample);
    const float rr_pdf = powf(kRestirPtRrProbability, static_cast<float>(sample.rr_count));
    return mul(mul(target_term, sample.radiance), rr_pdf);
}

__device__ bool restir_pt_surfaces_compatible_gpu(const GpuRestirSurface& current,
    const GpuRestirSurface& other, float expected_depth) {
    if (!current.valid || !other.valid || current.material != other.material ||
        current.object_id != other.object_id) return false;
    if (ddot(current.normal, other.normal) < kRestirPtNormalThreshold ||
        ddot(current.geo_normal, other.geo_normal) < kRestirPtNormalThreshold) return false;
    return fabsf(other.depth - expected_depth) <=
        fmaxf(0.01f, fabsf(expected_depth) * kRestirPtDepthThreshold);
}

__device__ void restir_pt_set_reconnection_gpu(GpuRestirPtPathState& state,
    Vec3 position, Vec3 normal, int rc_length, float wi_pdf) {
    if (state.rc_found) return;
    state.rc_found = 1;
    state.selected.position = position;
    state.selected.normal = normal;
    state.selected.rc_length = rc_length;
    state.selected.rc_wi_pdf = wi_pdf;
    state.selected.partial_jacobian = restir_pt_partial_inverse_jacobian_gpu(
        state.previous_position, position, normal);
    state.rc_throughput = state.throughput;
}

__device__ void restir_pt_stream_candidate_gpu(GpuRestirPtPathState& state,
    Vec3 estimator, int path_length, int fallback_rc_length, uint32_t flags,
    Vec3 fallback_position, Vec3 fallback_normal) {
    const float weight = restir_luminance_gpu(estimator);
    if (!(weight > 0.0f) || !isfinite(weight) || !finite_vec_gpu(estimator)) return;
    state.candidate_weight_sum += weight;
    if (rng_float(state.resampling_rng) * state.candidate_weight_sum >= weight) return;
    if (!state.rc_found) {
        state.selected.position = fallback_position;
        state.selected.normal = fallback_normal;
        state.selected.rc_length = fallback_rc_length;
        state.selected.partial_jacobian = (flags & GpuRestirPtEnvironment) != 0u ? 1.0f :
            restir_pt_partial_inverse_jacobian_gpu(state.previous_position,
                fallback_position, fallback_normal);
        state.selected.rc_wi_pdf = state.previous_pdf;
        state.rc_throughput = state.throughput;
    }
    const float rc_pdf = state.selected.rc_wi_pdf > 0.0f ? state.selected.rc_wi_pdf : 1.0f;
    const float rr_pdf = powf(kRestirPtRrProbability, static_cast<float>(state.rr_count));
    state.selected.target = mul(estimator, rc_pdf * rr_pdf);
    state.selected.radiance = restir_pt_safe_divide_gpu(estimator, state.rc_throughput);
    state.selected.path_length = path_length;
    state.selected.rr_count = state.rr_count;
    state.selected.flags = (state.rc_found ? 0u : flags) |
        (state.previous_delta ? GpuRestirPtDeltaPrefix : 0u);
}

__device__ void restir_pt_stream_nee_candidate_gpu(const GpuScene& scene,
    GpuRestirPtPathState& state, const GpuRestirReservoir& direct,
    Vec3 contribution, int path_length) {
    if (!direct.valid || !(direct.weight_sum > 0.0f)) return;
    const Vec3 estimator = mul(state.throughput, mul(contribution, direct.weight_sum));
    const float weight = restir_luminance_gpu(estimator);
    if (!(weight > 0.0f) || !isfinite(weight) || !finite_vec_gpu(estimator)) return;
    state.candidate_weight_sum += weight;
    if (rng_float(state.resampling_rng) * state.candidate_weight_sum >= weight) return;

    const float rr_pdf = powf(kRestirPtRrProbability, static_cast<float>(state.rr_count));
    if (!state.rc_found) {
        const float proposal_pdf = 1.0f / direct.weight_sum;
        state.selected.position = direct.sample.position_or_direction;
        state.selected.normal = direct.sample.type == GpuRestirLightType::Triangle &&
                direct.sample.index >= 0 && direct.sample.index < scene.triangle_count
            ? scene.triangles[direct.sample.index].normal : Vec3{};
        state.selected.radiance = restir_pt_pack_light_sample_gpu(direct.sample);
        state.selected.target = mul(estimator, proposal_pdf * rr_pdf);
        state.selected.partial_jacobian = proposal_pdf;
        state.selected.rc_wi_pdf = 0.0f;
        state.selected.rc_length = path_length;
        state.selected.flags = GpuRestirPtNeeLight |
            (state.previous_delta ? GpuRestirPtDeltaPrefix : 0u);
    } else {
        const float rc_pdf = state.selected.rc_wi_pdf > 0.0f ? state.selected.rc_wi_pdf : 1.0f;
        state.selected.target = mul(estimator, rc_pdf * rr_pdf);
        state.selected.radiance = restir_pt_safe_divide_gpu(estimator, state.rc_throughput);
        state.selected.flags = state.previous_delta ? GpuRestirPtDeltaPrefix : 0u;
    }
    state.selected.path_length = path_length;
    state.selected.rr_count = state.rr_count;
}

__device__ void restir_pt_finalize_path_reservoir_gpu(GpuRestirPtPathState& state) {
    state.selected.M = 1;
    state.selected.age = 0;
    const float selected_weight = restir_luminance_gpu(state.selected.target);
    state.selected.weight_sum = selected_weight > 0.0f
        ? state.candidate_weight_sum / selected_weight : 0.0f;
    if (!isfinite(state.selected.weight_sum)) state.selected.weight_sum = 0.0f;
    state.active = 0;
}

__global__ void restir_pt_reset_queue_kernel(GpuRestirPtQueueCounters* counters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        counters->active_count = 0;
        counters->next_count = 0;
    }
}

__global__ void restir_pt_promote_queue_kernel(GpuRestirPtQueueCounters* counters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        counters->active_count = counters->next_count;
        counters->next_count = 0;
    }
}

__global__ void restir_pt_setup_initial_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuWavefrontPaths paths, const GpuHit* primary_hits, const int* bsdf_indices,
    const GpuWavefrontQueueCounters* wavefront_counters, GpuRestirPtPathState* states,
    GpuPackedRestirPtReservoir* initial, GpuRestirSurface* surfaces,
    int* active_indices, GpuRestirPtQueueCounters* pt_counters, Vec3* samples,
    uint32_t sequence_index) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = wavefront_counters->num_queued[GpuWavefrontQueueBsdf];
    if (queue_index >= count) return;
    const int path_index = bsdf_indices[queue_index];
    GpuWavefrontPathRef path = wavefront_path_ref(paths, path_index);
    if (wavefront_bounce(path) - wavefront_transmission_bounces(path) != 0) return;
    const int pixel = path.pixel;
    initial[pixel] = {};
    surfaces[pixel] = {};
    const GpuScene& scene = *scene_ptr;
    const GpuHit hit = primary_hits[path_index];
    const GpuMaterial material = scene.materials[hit.material];
    const Vec3 wo = mul(path.ray.direction, -1.0f);
    const uint32_t replay_seed = make_pixel_seed(
        static_cast<uint32_t>(pixel % settings.width),
        static_cast<uint32_t>(pixel / settings.width),
        sequence_index ^ 0x6a09e667u);
    uint32_t replay_rng = replay_seed;
    const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal, wo,
        hit.uv, hit.front_face, replay_rng);
    if (!(sample.pdf > 0.0f) || !finite_vec_gpu(sample.weight)) {
        finish_wavefront_path(path, samples);
        return;
    }

    GpuRestirPtPathState state;
    state.path_index = path_index;
    state.replay_rng = replay_rng;
    state.resampling_rng = make_pixel_seed(static_cast<uint32_t>(pixel % settings.width),
        static_cast<uint32_t>(pixel / settings.width), sequence_index ^ 0xbb67ae85u);
    state.nee_rng = make_pixel_seed(static_cast<uint32_t>(pixel % settings.width),
        static_cast<uint32_t>(pixel / settings.width), sequence_index ^ 0x3f84d5b5u);
    state.selected.random_seed = replay_seed;
    state.selected.random_index = 0;
    state.throughput = sample.weight;
    state.previous_position = hit.position;
    state.previous_pdf = sample.pdf;
    state.previous_delta = sample.delta ? 1 : 0;
    restir_pt_reconnection_thresholds_gpu(replay_seed,
        state.footprint_threshold, state.pdf_threshold);
    state.previous_rough = !sample.delta && sample.pdf <= state.pdf_threshold ? 1 : 0;
    const float side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
    state.ray = {add(hit.position, mul(hit.normal, 0.001f * side)), sample.direction};
    state.depth = 1;
    state.active = 1;
    state.primary_surface.position = hit.position;
    state.primary_surface.normal = hit.normal;
    state.primary_surface.geo_normal = restir_geometric_normal_gpu(scene, hit);
    state.primary_surface.wo = wo;
    state.primary_surface.uv = hit.uv;
    state.primary_surface.material = hit.material;
    state.primary_surface.object_id = primary_object_id_gpu(hit);
    const Vec3 forward = dnormalize(sub(scene.camera.target, scene.camera.position));
    state.primary_surface.depth = fmaxf(0.0f,
        ddot(sub(hit.position, scene.camera.position), forward));
    state.primary_surface.valid = 1;
    states[pixel] = state;
    surfaces[pixel] = state.primary_surface;
    active_indices[atomicAdd(&pt_counters->active_count, 1)] = pixel;
}

template <bool AlphaVisibility, bool TwoLevel, GpuTraversalLayout Layout>
__global__ void restir_pt_trace_kernel(const GpuScene* scene_ptr,
    GpuRestirPtPathState* states, GpuCompactHit* compact_hits, int* trace_results,
    const int* active_indices, const GpuRestirPtQueueCounters* counters) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= counters->active_count) return;
    const int pixel = active_indices[queue_index];
    GpuRestirPtPathState& state = states[pixel];
    const GpuScene& scene = *scene_ptr;
    Ray ray = state.ray;
    for (int transparent_step = 0; transparent_step < 8; ++transparent_step) {
        GpuCompactHit compact{};
        if (!intersect_compact_gpu<TwoLevel, Layout>(scene, ray, compact)) {
            state.ray = ray;
            trace_results[pixel] = 2;
            return;
        }
        const int material_index = compact.triangle >= 0
            ? traversal_material_index(compact.material) : compact.material;
        if (material_index < 0 || material_index >= scene.material_count) {
            trace_results[pixel] = 0;
            return;
        }
        if constexpr (AlphaVisibility) {
            const GpuMaterial material = scene.materials[material_index];
            const Vec2 uv = compact_hit_uv_gpu(scene, ray, compact);
            if (!material_visible_gpu(scene, material, uv, state.replay_rng)) {
                const Vec3 position = compact_hit_position_gpu(ray, compact);
                ray = {add(position, mul(ray.direction, 0.002f)), ray.direction};
                continue;
            }
        }
        compact.material = material_index;
        state.ray = ray;
        compact_hits[pixel] = compact;
        trace_results[pixel] = 1;
        return;
    }
    trace_results[pixel] = 0;
}

__device__ bool restir_pt_should_record_bsdf_light_gpu(const RenderSettings& settings,
    int path_length, bool previous_delta) {
    if (previous_delta) return true;
    if (path_length == 2 && restir_direct_enabled_gpu(settings)) return false;
    return settings.sampling_mode != PathSamplingMode::NextEventEstimation;
}

__device__ bool restir_pt_should_record_bsdf_environment_gpu(const RenderSettings& settings,
    int path_length, bool previous_delta) {
    if (previous_delta) return true;
    return path_length != 2 || !restir_direct_enabled_gpu(settings);
}

__device__ float restir_pt_environment_bsdf_mis_weight_gpu(const GpuScene& scene,
    const RenderSettings& settings, const GpuRestirPtPathState& state, int path_length) {
    if (path_length == 2 || state.previous_delta ||
        settings.sampling_mode != PathSamplingMode::MultipleImportanceSampling) return 1.0f;
    GpuRestirLightSample sample;
    sample.type = GpuRestirLightType::Environment;
    sample.position_or_direction = state.ray.direction;
    sample.uv = restir_environment_uv_from_direction_gpu(scene, state.ray.direction);
    sample.index = restir_environment_texel_from_uv_gpu(scene, sample.uv);
    GpuRestirSurface surface;
    surface.position = state.previous_position;
    const float light_pdf = restir_pt_light_proposal_pdf_gpu(scene, settings, surface, sample);
    return mis_weight_gpu(state.previous_pdf, light_pdf,
        static_cast<int>(settings.mis_heuristic));
}

__global__ void restir_pt_resolve_trace_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuRestirPtPathState* states, const GpuCompactHit* compact_hits, GpuHit* hits,
    int* trace_results, const int* active_indices, const GpuRestirPtQueueCounters* counters) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= counters->active_count) return;
    const int pixel = active_indices[queue_index];
    GpuRestirPtPathState& state = states[pixel];
    const GpuScene& scene = *scene_ptr;
    const int path_length = state.depth + 1;
    if (trace_results[pixel] == 2) {
        if (restir_pt_should_record_bsdf_environment_gpu(
                settings, path_length, state.previous_delta != 0)) {
            const float mis = restir_pt_environment_bsdf_mis_weight_gpu(
                scene, settings, state, path_length);
            Vec3 target = mul(state.throughput, mul(
                environment_radiance_gpu(scene, state.ray.direction, settings), mis));
            restir_pt_stream_candidate_gpu(state, target, path_length, path_length,
                GpuRestirPtEnvironment, state.ray.direction, mul(state.ray.direction, -1.0f));
        }
        restir_pt_finalize_path_reservoir_gpu(state);
        return;
    }
    GpuHit hit;
    if (trace_results[pixel] != 1 ||
        !fill_compact_hit_gpu(scene, state.ray, compact_hits[pixel], hit)) {
        restir_pt_finalize_path_reservoir_gpu(state);
        return;
    }
    GpuMaterial material = scene.materials[hit.material];
    hit.normal = apply_normal_map_gpu(scene, material, hit, state.ray.direction);
    const Vec3 material_emission = material_emission_gpu(scene, material, hit.uv, settings);
    hit.emission = add(hit.emission, material_emission);

    const float distance2 = fmaxf(1.0e-10f,
        ddot(sub(hit.position, state.previous_position), sub(hit.position, state.previous_position)));
    const float geometry = fabsf(ddot(hit.normal, mul(state.ray.direction, -1.0f))) / distance2;
    const float footprint = 1.0f / fmaxf(1.0e-12f, geometry * state.previous_pdf);
    state.previous_far = settings.cuda_restir_pt_reconnection == RestirPtReconnectionMode::FixedThreshold
        ? (sqrtf(distance2) > 0.0f && !state.previous_delta)
        : (footprint > state.footprint_threshold && !state.previous_delta);
    const bool current_rough = settings.cuda_restir_pt_reconnection == RestirPtReconnectionMode::FixedThreshold
        ? restir_pt_material_roughness_gpu(scene, material, hit.uv) > 0.1f
        : restir_pt_material_roughness_gpu(scene, material, hit.uv) > 0.02f;
    if (!state.rc_found && state.previous_rough && state.previous_far && current_rough) {
        restir_pt_set_reconnection_gpu(state, hit.position, hit.normal, path_length, state.previous_pdf);
    }

    if (has_light_emission_gpu(hit.emission)) {
        const GpuTriangle light = hit.triangle >= 0 && hit.triangle < scene.triangle_count
            ? scene.triangles[hit.triangle] : GpuTriangle{};
        const Vec3 emission = emitted_radiance_gpu(light, material, light.emission,
            material_emission, light.light_double_sided != 0, state.ray.direction);
        if (restir_pt_should_record_bsdf_light_gpu(settings, path_length, state.previous_delta != 0)) {
            float mis = 1.0f;
            if (!state.previous_delta && settings.sampling_mode == PathSamplingMode::MultipleImportanceSampling &&
                hit.triangle >= 0 && scene.light_count > 0) {
                GpuRestirLightSample light_sample;
                light_sample.type = GpuRestirLightType::Triangle;
                light_sample.index = hit.triangle;
                light_sample.position_or_direction = hit.position;
                light_sample.uv = hit.uv;
                light_sample.sampler_index = -1;
                for (int i = 0; i < scene.light_count; ++i) {
                    if (scene.light_indices[i] == hit.triangle) {
                        light_sample.sampler_index = i;
                        break;
                    }
                }
                GpuRestirSurface previous_surface;
                previous_surface.position = state.previous_position;
                const float light_pdf = restir_pt_light_proposal_pdf_gpu(
                    scene, settings, previous_surface, light_sample);
                mis = mis_weight_gpu(state.previous_pdf, light_pdf,
                    static_cast<int>(settings.mis_heuristic));
            }
            restir_pt_stream_candidate_gpu(state, mul(state.throughput, mul(emission, mis)),
                path_length, path_length, 0u, hit.position, hit.normal);
        }
        restir_pt_finalize_path_reservoir_gpu(state);
        return;
    }
    state.hit = hit;
    hits[pixel] = hit;
    trace_results[pixel] = 3;
}

template <bool HasLocalLights, bool HasEnvironment>
__global__ void restir_pt_generate_nee_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    GpuRestirPtPathState* states, const GpuHit* hits, const int* trace_results,
    const int* active_indices, const GpuRestirPtQueueCounters* counters,
    GpuRestirReservoir* nee_reservoirs, GpuRestirVisibilityRay* visibility_rays,
    int* visibility_results, int* visibility_indices, int* visibility_count) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= counters->active_count) return;
    const int pixel = active_indices[queue_index];
    if (trace_results[pixel] != 3) return;
    GpuRestirPtPathState& state = states[pixel];
    const GpuScene& scene = *scene_ptr;
    const GpuHit hit = hits[pixel];
    const GpuMaterial material = scene.materials[hit.material];
    const Vec3 wo = mul(state.ray.direction, -1.0f);
    GpuRestirReservoir direct = restir_pt_sample_nee_gpu<HasLocalLights, HasEnvironment>(
        scene, settings, hit, material, wo, state.nee_rng);
    nee_reservoirs[pixel] = direct;
    visibility_rays[pixel] = {};
    visibility_results[pixel] = -1;
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
    visibility.path_index = state.path_index;
    visibility.valid = 1;
    visibility_rays[pixel] = visibility;
    visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
}

__global__ void restir_pt_stream_nee_continue_kernel(const GpuScene* scene_ptr,
    RenderSettings settings, GpuRestirPtPathState* states, const GpuHit* hits,
    const int* trace_results, const GpuRestirReservoir* nee_reservoirs,
    const int* visibility_results, const int* active_indices,
    int* next_indices, GpuRestirPtQueueCounters* counters) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= counters->active_count) return;
    const int pixel = active_indices[queue_index];
    GpuRestirPtPathState& state = states[pixel];
    if (!state.active || trace_results[pixel] != 3) return;
    const GpuScene& scene = *scene_ptr;
    const GpuHit hit = hits[pixel];
    const GpuMaterial material = scene.materials[hit.material];
    const Vec3 wo = mul(state.ray.direction, -1.0f);
    if (visibility_results[pixel] > 0) {
        const GpuRestirReservoir direct = nee_reservoirs[pixel];
        if (direct.valid && direct.weight_sum > 0.0f) {
            Vec3 direction; float distance = 0.0f;
            Vec3 contribution = restir_evaluate_sample_gpu(scene, hit, material, wo,
                settings, direct.sample, direction, distance);
            const float proposal_pdf = 1.0f / direct.weight_sum;
            contribution = mul(contribution, restir_pt_nee_mis_weight_gpu(
                scene, settings, hit, material, wo, direction, proposal_pdf));
            restir_pt_stream_nee_candidate_gpu(scene, state, direct, contribution,
                state.depth + 2);
        }
    }

    const int max_depth = iclamp_gpu(settings.cuda_restir_pt_max_bounces, 2,
        iclamp_gpu(settings.max_bounces, 2, 8));
    const GpuMaterialSample sample = sample_material_gpu(scene, material, hit.normal,
        wo, hit.uv, hit.front_face, state.replay_rng);
    if (!(sample.pdf > 0.0f) || !finite_vec_gpu(sample.weight)) {
        restir_pt_finalize_path_reservoir_gpu(state);
        return;
    }
    const bool beyond_regular_depth = state.depth + 1 >= max_depth;
    if (beyond_regular_depth && (!sample.delta || state.depth + 1 >= max_depth + kRestirPtExtraDeltaBudget)) {
        restir_pt_finalize_path_reservoir_gpu(state);
        return;
    }
    Vec3 next_throughput = mul(state.throughput, sample.weight);
    if (state.depth >= 1) {
        uint32_t rr_rng = restir_rng_seed_gpu(state.selected.random_seed,
            0xa4093822u ^ static_cast<uint32_t>(state.depth) * 0x9e3779b9u);
        if (max_channel_gpu(next_throughput) < kRestirPtThroughputCutoff ||
            rng_float(rr_rng) >= kRestirPtRrProbability) {
            restir_pt_finalize_path_reservoir_gpu(state);
            return;
        }
        next_throughput = divv(next_throughput, kRestirPtRrProbability);
        ++state.rr_count;
    }
    state.previous_position = hit.position;
    state.previous_pdf = sample.pdf;
    state.previous_delta = sample.delta ? 1 : 0;
    state.previous_rough = !sample.delta && sample.pdf <= state.pdf_threshold ? 1 : 0;
    state.throughput = next_throughput;
    const float side = ddot(sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
    state.ray = {add(hit.position, mul(hit.normal, 0.001f * side)), sample.direction};
    ++state.depth;
    next_indices[atomicAdd(&counters->next_count, 1)] = pixel;
}

__global__ void restir_pt_finalize_initial_kernel(GpuRestirPtPathState* states,
    GpuPackedRestirPtReservoir* initial, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    GpuRestirPtPathState& state = states[pixel];
    if (state.path_index < 0) return;
    if (state.active) restir_pt_finalize_path_reservoir_gpu(state);
    initial[pixel] = restir_pt_pack_gpu(state.selected);
}

__device__ bool restir_pt_generate_nee_light_ray_gpu(const GpuScene& scene,
    const RenderSettings& settings, const GpuRestirSurface& surface,
    const GpuRestirPtReservoir& reservoir, int path_index,
    GpuRestirVisibilityRay& visibility, Vec3& contribution, float& proposal_pdf) {
    const GpuRestirLightSample sample = restir_pt_unpack_light_sample_gpu(reservoir);
    Vec3 direction;
    float distance = 0.0f;
    contribution = restir_evaluate_surface_sample_gpu(
        scene, surface, settings, sample, direction, distance);
    proposal_pdf = restir_pt_light_proposal_pdf_gpu(scene, settings, surface, sample);
    contribution = mul(contribution, restir_pt_nee_mis_weight_gpu(
        scene, settings, surface, direction, proposal_pdf));
    if (!(restir_luminance_gpu(contribution) > 0.0f) ||
        !(proposal_pdf > 0.0f) || !isfinite(proposal_pdf)) return false;
    const float side = ddot(surface.geo_normal, direction) >= 0.0f ? 1.0f : -1.0f;
    visibility.ray = {add(surface.position, mul(surface.geo_normal, 0.002f * side)), direction};
    visibility.remaining = isfinite(distance) ? distance - 0.01f : kInfinity;
    visibility.sample_type = static_cast<int>(sample.type);
    visibility.sample_index = sample.index;
    visibility.path_index = path_index;
    visibility.valid = 1;
    return true;
}

__device__ bool restir_pt_generate_connection_ray_gpu(const GpuRestirSurface& surface,
    const GpuRestirPtReservoir& reservoir, int path_index, GpuRestirVisibilityRay& visibility) {
    Vec3 direction;
    float remaining = kInfinity;
    if ((reservoir.flags & GpuRestirPtEnvironment) != 0u) {
        direction = dnormalize(reservoir.position);
    } else {
        const Vec3 delta = sub(reservoir.position, surface.position);
        const float distance = sqrtf(ddot(delta, delta));
        if (!(distance > 0.003f)) return false;
        direction = divv(delta, distance);
        if (!(ddot(reservoir.normal, mul(direction, -1.0f)) > 1.0e-6f)) return false;
        remaining = distance - 0.01f;
    }
    const float side = ddot(surface.geo_normal, direction) >= 0.0f ? 1.0f : -1.0f;
    visibility.ray = {add(surface.position, mul(surface.geo_normal, 0.002f * side)), direction};
    visibility.remaining = remaining;
    visibility.sample_type = static_cast<int>(GpuRestirLightType::Environment);
    visibility.sample_index = -1;
    visibility.path_index = path_index;
    visibility.valid = 1;
    return true;
}

__device__ int2 restir_pt_temporal_offset_gpu(int sample_index) {
    sample_index &= 7;
    const float angle = (static_cast<float>(sample_index) + 0.5f) * (2.0f * kPi / 8.0f);
    return make_int2(static_cast<int>(roundf(cosf(angle))), static_cast<int>(roundf(sinf(angle))));
}

__global__ void restir_pt_temporal_prepare_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    const GpuRestirSurface* current_surfaces, const GpuRestirSurface* history_surfaces,
    const GpuPackedRestirPtReservoir* history, const uint32_t* duplication_counts,
    GpuRestirPtTemporalState* states,
    GpuRestirVisibilityRay* rays, int* results, int* visibility_indices, int* visibility_count,
    const GpuRestirPtPathState* path_states, Camera history_camera, bool history_valid,
    uint32_t sequence_index, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    states[pixel] = {};
    rays[pixel] = {};
    results[pixel] = -1;
    const GpuRestirSurface surface = current_surfaces[pixel];
    if (!surface.valid || !history_valid ||
        settings.cuda_restir_pt_resampling == RestirPtResamplingMode::None) return;
    float projected_x = 0.0f, projected_y = 0.0f, expected_depth = 0.0f;
    if (!restir_project_to_camera_gpu(history_camera, surface.position, settings.width,
            settings.height, projected_x, projected_y, expected_depth)) return;
    const int px = static_cast<int>(roundf(projected_x));
    const int py = static_cast<int>(roundf(projected_y));
    uint32_t rng = make_pixel_seed(static_cast<uint32_t>(pixel % settings.width),
        static_cast<uint32_t>(pixel / settings.width), sequence_index ^ 0x3c6ef372u);
    const int start = static_cast<int>(rng_float(rng) * 8.0f);
    int best_pixel = -1;
    for (int i = 0; i < 5; ++i) {
        const int2 offset = i == 0 ? make_int2(0, 0) : restir_pt_temporal_offset_gpu(start + i);
        const int x = px + offset.x;
        const int y = py + offset.y;
        if (x < 0 || y < 0 || x >= settings.width || y >= settings.height) continue;
        const int candidate = y * settings.width + x;
        if (!restir_pt_surfaces_compatible_gpu(surface, history_surfaces[candidate], expected_depth)) continue;
        const GpuRestirPtReservoir reservoir = restir_pt_unpack_gpu(history[candidate]);
        if (!restir_pt_valid_gpu(reservoir) || reservoir.age >= kRestirPtMaxAge) continue;
        best_pixel = candidate;
        break;
    }
    if (best_pixel < 0 && restir_pt_surfaces_compatible_gpu(surface,
            history_surfaces[pixel], surface.depth)) {
        const GpuRestirPtReservoir fallback = restir_pt_unpack_gpu(history[pixel]);
        if (restir_pt_valid_gpu(fallback) && fallback.age < kRestirPtMaxAge) best_pixel = pixel;
    }
    if (best_pixel < 0) return;
    GpuRestirPtReservoir source = restir_pt_unpack_gpu(history[best_pixel]);
    GpuRestirPtTemporalState state;
    state.source_pixel = best_pixel;
    const int max_history = duplication_counts != nullptr
        ? restir_pt_reduced_history_length_gpu(duplication_counts[best_pixel])
        : kRestirPtMaxHistoryLength;
    state.source_m = source.M < max_history ? source.M : max_history;
    state.valid = 1;
    if (source.rc_length <= 2) {
        if (!restir_pt_validate_reconnection_gpu(*scene_ptr, settings, surface, source)) return;
        const float jacobian = restir_pt_reconnection_jacobian_gpu(surface, source);
        const Vec3 shifted_target = restir_pt_shift_target_gpu(*scene_ptr, surface,
            history_surfaces[best_pixel], source);
        if (!(jacobian > 0.0f) || !(restir_luminance_gpu(shifted_target) > 0.0f)) return;
        GpuRestirVisibilityRay visibility;
        if (!restir_pt_generate_connection_ray_gpu(surface, source,
                path_states[pixel].path_index, visibility)) return;
        state.jacobian = jacobian;
        rays[pixel] = visibility;
        visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
    }
    states[pixel] = state;
}

template <typename ReuseState>
__global__ void restir_pt_replay_setup_kernel(const GpuScene* scene_ptr,
    GpuRestirPtPathState* path_states, ReuseState* reuse_states,
    const GpuPackedRestirPtReservoir* source_reservoirs,
    const GpuRestirSurface* receiver_surfaces,
    int* active_indices, GpuRestirPtQueueCounters* counters, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    ReuseState& reuse = reuse_states[pixel];
    if (!reuse.valid || reuse.source_pixel < 0) return;
    const GpuRestirPtReservoir source = restir_pt_unpack_gpu(source_reservoirs[reuse.source_pixel]);
    if (!restir_pt_valid_gpu(source) || source.rc_length <= 2) return;
    GpuRestirPtPathState state;
    state.path_index = path_states[pixel].path_index;
    state.primary_surface = receiver_surfaces[pixel];
    state.selected = source;
    state.replay_rng = source.random_seed;
    state.resampling_rng = restir_rng_seed_gpu(source.random_seed, 0xcbbb9d5du);
    const GpuScene& scene = *scene_ptr;
    const GpuRestirSurface surface = state.primary_surface;
    if (!surface.valid || surface.material < 0 || surface.material >= scene.material_count) {
        reuse.valid = 0;
        return;
    }
    const GpuMaterial material = scene.materials[surface.material];
    const bool front_face = ddot(surface.geo_normal, surface.wo) >= 0.0f;
    const GpuMaterialSample sample = sample_material_gpu(scene, material, surface.normal,
        surface.wo, surface.uv, front_face, state.replay_rng);
    if (!(sample.pdf > 0.0f) || !finite_vec_gpu(sample.weight)) {
        reuse.valid = 0;
        return;
    }
    state.throughput = sample.weight;
    state.previous_position = surface.position;
    state.previous_pdf = sample.pdf;
    state.previous_delta = sample.delta ? 1 : 0;
    state.depth = 1;
    state.active = 1;
    const float side = ddot(sample.direction, surface.normal) >= 0.0f ? 1.0f : -1.0f;
    state.ray = {add(surface.position, mul(surface.normal, 0.001f * side)), sample.direction};
    path_states[pixel] = state;
    active_indices[atomicAdd(&counters->active_count, 1)] = pixel;
}

template <typename ReuseState>
__global__ void restir_pt_inverse_prepare_kernel(const GpuScene* scene_ptr,
    RenderSettings settings,
    const GpuRestirSurface* receiver_surfaces,
    const GpuPackedRestirPtReservoir* selected_reservoirs, ReuseState* reuse_states,
    const GpuRestirPtPathState* path_states, GpuRestirVisibilityRay* rays, int* results,
    int* visibility_indices, int* visibility_count, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    rays[pixel] = {};
    results[pixel] = -1;
    ReuseState& reuse = reuse_states[pixel];
    reuse.replayed = 0;
    reuse.jacobian = 0.0f;
    if (reuse.source_pixel < 0) return;
    reuse.valid = 1;
    const GpuRestirSurface receiver = receiver_surfaces[reuse.source_pixel];
    const GpuRestirPtReservoir selected = restir_pt_unpack_gpu(selected_reservoirs[pixel]);
    if (!receiver.valid || !restir_pt_valid_gpu(selected) || selected.rc_length > 2) return;
    if (!restir_pt_validate_reconnection_gpu(*scene_ptr, settings, receiver, selected)) return;
    const float jacobian = restir_pt_reconnection_jacobian_gpu(receiver, selected);
    const Vec3 target = restir_pt_shift_target_gpu(*scene_ptr, receiver, receiver, selected);
    if (!(jacobian > 0.0f) || !(restir_luminance_gpu(target) > 0.0f)) return;
    GpuRestirVisibilityRay visibility;
    if (!restir_pt_generate_connection_ray_gpu(receiver, selected,
            path_states[pixel].path_index, visibility)) return;
    reuse.jacobian = jacobian;
    rays[pixel] = visibility;
    visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
}

template <typename ReuseState>
__global__ void restir_pt_inverse_replay_setup_kernel(const GpuScene* scene_ptr,
    GpuRestirPtPathState* path_states, ReuseState* reuse_states,
    const GpuPackedRestirPtReservoir* selected_reservoirs,
    const GpuRestirSurface* receiver_surfaces, int* active_indices,
    GpuRestirPtQueueCounters* counters, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    ReuseState& reuse = reuse_states[pixel];
    if (reuse.source_pixel < 0) return;
    reuse.valid = 1;
    const GpuRestirPtReservoir selected = restir_pt_unpack_gpu(selected_reservoirs[pixel]);
    if (!restir_pt_valid_gpu(selected) || selected.rc_length <= 2) return;
    const GpuRestirSurface surface = receiver_surfaces[reuse.source_pixel];
    if (!surface.valid || surface.material < 0 || surface.material >= scene_ptr->material_count) return;

    GpuRestirPtPathState state;
    state.path_index = path_states[pixel].path_index;
    state.primary_surface = surface;
    state.selected = selected;
    state.replay_rng = selected.random_seed;
    state.resampling_rng = restir_rng_seed_gpu(selected.random_seed, 0x4a7484aau);
    const GpuScene& scene = *scene_ptr;
    const GpuMaterial material = scene.materials[surface.material];
    const bool front_face = ddot(surface.geo_normal, surface.wo) >= 0.0f;
    const GpuMaterialSample sample = sample_material_gpu(scene, material, surface.normal,
        surface.wo, surface.uv, front_face, state.replay_rng);
    if (!(sample.pdf > 0.0f) || !finite_vec_gpu(sample.weight)) return;
    state.throughput = sample.weight;
    state.previous_position = surface.position;
    state.previous_pdf = sample.pdf;
    state.previous_delta = sample.delta ? 1 : 0;
    state.depth = 1;
    state.active = 1;
    const float side = ddot(sample.direction, surface.normal) >= 0.0f ? 1.0f : -1.0f;
    state.ray = {add(surface.position, mul(surface.normal, 0.001f * side)), sample.direction};
    path_states[pixel] = state;
    active_indices[atomicAdd(&counters->active_count, 1)] = pixel;
}

template <typename ReuseState>
__global__ void restir_pt_replay_resolve_kernel(const GpuScene* scene_ptr,
    RenderSettings settings,
    GpuRestirPtPathState* path_states, ReuseState* reuse_states,
    const GpuCompactHit* compact_hits, const int* trace_results,
    const int* active_indices, int* next_indices, GpuRestirPtQueueCounters* counters,
    GpuPackedRestirPtReservoir* shifted_reservoirs,
    GpuRestirVisibilityRay* visibility_rays, int* visibility_results,
    int* visibility_indices, int* visibility_count) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= counters->active_count) return;
    const int pixel = active_indices[queue_index];
    GpuRestirPtPathState& state = path_states[pixel];
    ReuseState& reuse = reuse_states[pixel];
    if (!reuse.valid || !state.active || trace_results[pixel] != 1) {
        reuse.valid = 0;
        state.active = 0;
        return;
    }
    const GpuScene& scene = *scene_ptr;
    GpuHit hit;
    if (!fill_compact_hit_gpu(scene, state.ray, compact_hits[pixel], hit)) {
        reuse.valid = 0;
        state.active = 0;
        return;
    }
    GpuMaterial material = scene.materials[hit.material];
    hit.normal = apply_normal_map_gpu(scene, material, hit, state.ray.direction);
    const int vertex_length = state.depth + 1;
    const Vec3 wo = mul(state.ray.direction, -1.0f);
    const GpuMaterialSample replay_sample = sample_material_gpu(scene, material, hit.normal,
        wo, hit.uv, hit.front_face, state.replay_rng);
    if (!(replay_sample.pdf > 0.0f) || !finite_vec_gpu(replay_sample.weight)) {
        reuse.valid = 0;
        state.active = 0;
        return;
    }

    if (vertex_length + 1 == state.selected.rc_length) {
        GpuRestirSurface connection_surface{hit.position, hit.normal,
            restir_geometric_normal_gpu(scene, hit), wo, hit.uv, 0.0f,
            primary_object_id_gpu(hit), hit.material, 1};
        if ((state.selected.flags & GpuRestirPtNeeLight) != 0u) {
            Vec3 light_contribution;
            float proposal_pdf = 0.0f;
            GpuRestirVisibilityRay visibility;
            if (!restir_pt_generate_nee_light_ray_gpu(scene, settings, connection_surface,
                    state.selected, state.path_index, visibility,
                    light_contribution, proposal_pdf)) {
                reuse.valid = 0;
                state.active = 0;
                return;
            }
            const float rr_pdf = powf(kRestirPtRrProbability,
                static_cast<float>(state.selected.rr_count));
            const Vec3 target = mul(mul(state.throughput, light_contribution), rr_pdf);
            const float jacobian = proposal_pdf > 0.0f &&
                    state.selected.partial_jacobian > 0.0f
                ? state.selected.partial_jacobian / proposal_pdf : 0.0f;
            if (!(jacobian > 0.0f) || !isfinite(jacobian) ||
                !(restir_luminance_gpu(target) > 0.0f)) {
                reuse.valid = 0;
                state.active = 0;
                return;
            }
            state.selected.target = target;
            state.selected.partial_jacobian = proposal_pdf;
            shifted_reservoirs[pixel] = restir_pt_pack_gpu(state.selected);
            reuse.replayed = 1;
            reuse.jacobian = jacobian;
            visibility_rays[pixel] = visibility;
            visibility_results[pixel] = -1;
            visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
            state.active = 0;
            return;
        }

        const Vec3 delta = sub(state.selected.position, hit.position);
        const float distance2 = ddot(delta, delta);
        if (!(distance2 > 1.0e-10f)) { reuse.valid = 0; state.active = 0; return; }
        const Vec3 wi = mul(delta, rsqrtf(distance2));
        if (!restir_gi_receiver_accepts_direction_gpu(connection_surface, material, wi) ||
            !restir_pt_validate_reconnection_gpu(scene, settings,
                connection_surface, state.selected)) {
            reuse.valid = 0;
            state.active = 0;
            return;
        }
        const float cosine = material.double_sided ||
            material.brdf_model == static_cast<int>(BrdfModel::DiffuseTransmission)
            ? fabsf(ddot(hit.normal, wi)) : fmaxf(0.0f, ddot(hit.normal, wi));
        const Vec3 connection = mul(evaluate_brdf_gpu(scene, material, hit.normal,
            wo, wi, hit.uv), cosine / replay_sample.pdf);
        const Vec3 estimator = mul(state.throughput, mul(connection, state.selected.radiance));
        const float rr_pdf = powf(kRestirPtRrProbability,
            static_cast<float>(state.selected.rr_count));
        const Vec3 target = mul(estimator, replay_sample.pdf * rr_pdf);
        const float new_inverse = restir_pt_partial_inverse_jacobian_gpu(
            hit.position, state.selected.position, state.selected.normal);
        const float jacobian = new_inverse > 0.0f && state.selected.partial_jacobian > 0.0f
            ? state.selected.partial_jacobian / new_inverse : 0.0f;
        if (!(jacobian >= 0.1f && jacobian <= 10.0f) ||
            !(restir_luminance_gpu(target) > 0.0f)) {
            reuse.valid = 0;
            state.active = 0;
            return;
        }
        state.selected.target = target;
        state.selected.partial_jacobian = new_inverse;
        state.selected.rc_wi_pdf = replay_sample.pdf;
        shifted_reservoirs[pixel] = restir_pt_pack_gpu(state.selected);
        GpuRestirVisibilityRay visibility;
        if (!restir_pt_generate_connection_ray_gpu(connection_surface, state.selected,
                state.path_index, visibility)) {
            reuse.valid = 0;
            state.active = 0;
            return;
        }
        reuse.replayed = 1;
        reuse.jacobian = jacobian;
        visibility_rays[pixel] = visibility;
        visibility_results[pixel] = -1;
        visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
        state.active = 0;
        return;
    }

    Vec3 throughput = mul(state.throughput, replay_sample.weight);
    if (state.depth >= 1) {
        // Replay is conditioned on the source path having survived roulette. Keep
        // its PDF compensation, but do not introduce a new failure or consume a
        // BSDF replay dimension.
        throughput = divv(throughput, kRestirPtRrProbability);
        ++state.rr_count;
    }
    state.throughput = throughput;
    state.previous_position = hit.position;
    state.previous_pdf = replay_sample.pdf;
    const float side = ddot(replay_sample.direction, hit.normal) >= 0.0f ? 1.0f : -1.0f;
    state.ray = {add(hit.position, mul(hit.normal, 0.001f * side)), replay_sample.direction};
    ++state.depth;
    next_indices[atomicAdd(&counters->next_count, 1)] = pixel;
}

__device__ bool restir_pt_combine_gpu(GpuRestirPtReservoir& output,
    const GpuRestirPtReservoir& candidate, Vec3 target, float jacobian,
    int candidate_m, float random) {
    const int total_m = output.M + candidate_m;
    output.M = total_m;
    const float weight = restir_luminance_gpu(target) * jacobian *
        candidate.weight_sum * static_cast<float>(candidate_m);
    if (!(weight > 0.0f) || !isfinite(weight)) return false;
    output.weight_sum += weight;
    if (random * output.weight_sum >= weight) return false;
    const float accumulated_weight = output.weight_sum;
    output = candidate;
    output.target = target;
    output.weight_sum = accumulated_weight;
    output.M = total_m;
    return true;
}

__device__ float restir_pt_inverse_pairwise_density_gpu(const GpuScene& scene,
    const GpuRestirSurface& source_surface, const GpuRestirPtReservoir& selected) {
    if (selected.rc_length > 2) return 0.0f;
    const Vec3 inverse_target = restir_pt_shift_target_gpu(scene, source_surface,
        source_surface, selected);
    const float inverse_jacobian = restir_pt_reconnection_jacobian_gpu(source_surface, selected);
    const float density = restir_luminance_gpu(inverse_target) * inverse_jacobian;
    return isfinite(density) ? density : 0.0f;
}

__global__ void restir_pt_temporal_select_kernel(const GpuScene* scene_ptr,
    const GpuRestirSurface* current_surfaces, const GpuRestirSurface* history_surfaces,
    const GpuPackedRestirPtReservoir* initial, const GpuPackedRestirPtReservoir* history,
    const GpuPackedRestirPtReservoir* shifted,
    GpuRestirPtTemporalState* states, const int* visibility_results,
    GpuPackedRestirPtReservoir* output, uint32_t sequence_index, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirSurface current_surface = current_surfaces[pixel];
    GpuRestirPtReservoir canonical = restir_pt_unpack_gpu(initial[pixel]);
    if (!current_surface.valid) { output[pixel] = {}; return; }
    GpuRestirPtReservoir result;
    uint32_t rng = make_pixel_seed(static_cast<uint32_t>(pixel), sequence_index, 0xa54ff53au);
    bool selected_source = false;
    if (canonical.M > 0) {
        restir_pt_combine_gpu(result, canonical, canonical.target, 1.0f,
            canonical.M, rng_float(rng));
    }
    GpuRestirPtTemporalState state = states[pixel];
    GpuRestirPtReservoir source;
    Vec3 source_target{};
    float source_jacobian = 0.0f;
    if (state.valid) {
        source = state.replayed != 0
            ? restir_pt_unpack_gpu(shifted[pixel])
            : restir_pt_unpack_gpu(history[state.source_pixel]);
        source.M = state.source_m;
        if (visibility_results[pixel] > 0) {
            source_target = state.replayed != 0 ? source.target :
                restir_pt_shift_target_gpu(*scene_ptr, current_surface,
                    history_surfaces[state.source_pixel], source);
        }
        source_jacobian = state.jacobian;
        selected_source = restir_pt_combine_gpu(result, source, source_target,
            source_jacobian, source.M, rng_float(rng));
    }
    const int total_m = result.M;
    result.M = total_m;
    if (selected_source) {
        result.age = source.age + 1;
        if (result.rc_length <= 2) {
            result.partial_jacobian = restir_pt_partial_inverse_jacobian_gpu(
                current_surface.position, result.position, result.normal);
        }
    }
    state.selected_source = selected_source ? 1 : 0;
    states[pixel] = state;
    output[pixel] = restir_pt_pack_gpu(result);
}

template <typename ReuseState>
__global__ void restir_pt_pairwise_normalize_kernel(const GpuScene* scene_ptr,
    const GpuRestirSurface* inverse_receiver_surfaces,
    const GpuPackedRestirPtReservoir* canonical_reservoirs,
    const GpuPackedRestirPtReservoir* provisional_reservoirs,
    const GpuPackedRestirPtReservoir* inverse_shifted_reservoirs,
    const ReuseState* states, const int* visibility_results,
    GpuPackedRestirPtReservoir* output, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirPtReservoir canonical = restir_pt_unpack_gpu(canonical_reservoirs[pixel]);
    GpuRestirPtReservoir selected = restir_pt_unpack_gpu(provisional_reservoirs[pixel]);
    if (!restir_pt_valid_gpu(selected)) { output[pixel] = {}; return; }
    const ReuseState state = states[pixel];
    const float selected_target = restir_luminance_gpu(selected.target);
    float pi = selected_target;
    float pi_sum = selected_target * static_cast<float>(canonical.M);
    if (state.source_pixel >= 0 && state.source_m > 0) {
        float inverse_density = 0.0f;
        if (visibility_results[pixel] > 0 && state.jacobian > 0.0f) {
            Vec3 inverse_target;
            if (state.replayed != 0) {
                inverse_target = restir_pt_unpack_gpu(inverse_shifted_reservoirs[pixel]).target;
            } else {
                const GpuRestirSurface receiver = inverse_receiver_surfaces[state.source_pixel];
                inverse_target = restir_pt_shift_target_gpu(*scene_ptr, receiver, receiver, selected);
            }
            inverse_density = restir_luminance_gpu(inverse_target) * state.jacobian;
            if (!isfinite(inverse_density)) inverse_density = 0.0f;
        }
        if (state.selected_source != 0) pi = inverse_density;
        pi_sum += inverse_density * static_cast<float>(state.source_m);
    }
    selected.weight_sum = selected_target > 0.0f && pi_sum > 0.0f
        ? selected.weight_sum * pi / (selected_target * pi_sum) : 0.0f;
    output[pixel] = restir_pt_pack_gpu(selected);
}

__global__ void restir_pt_boiling_filter_kernel(const GpuPackedRestirPtReservoir* input,
    GpuPackedRestirPtReservoir* output, int width, int height) {
    __shared__ float values[256];
    __shared__ int counts[256];
    const int local = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const bool inside = x < width && y < height;
    const int pixel = y * width + x;
    const GpuRestirPtReservoir reservoir = inside ? restir_pt_unpack_gpu(input[pixel]) : GpuRestirPtReservoir{};
    const float value = restir_pt_valid_gpu(reservoir)
        ? restir_luminance_gpu(reservoir.target) * reservoir.weight_sum : 0.0f;
    values[local] = value;
    counts[local] = value > 0.0f ? 1 : 0;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (local < stride) { values[local] += values[local + stride]; counts[local] += counts[local + stride]; }
        __syncthreads();
    }
    if (!inside) return;
    const float average = counts[0] > 0 ? values[0] / static_cast<float>(counts[0]) : 0.0f;
    const float multiplier = 10.0f / kRestirPtBoilingStrength - 9.0f;
    output[pixel] = value > average * multiplier ? GpuPackedRestirPtReservoir{} : input[pixel];
}

__global__ void restir_pt_spatial_prepare_kernel(const GpuScene* scene_ptr, RenderSettings settings,
    const GpuRestirSurface* surfaces, const GpuPackedRestirPtReservoir* input,
    const int* pairing_maps,
    GpuRestirPtSpatialState* states, GpuRestirVisibilityRay* rays, int* results,
    int* visibility_indices, int* visibility_count, const GpuRestirPtPathState* path_states,
    uint32_t sequence_index, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    states[pixel] = {};
    rays[pixel] = {};
    results[pixel] = -1;
    const GpuRestirSurface surface = surfaces[pixel];
    if (!surface.valid || pairing_maps == nullptr) return;
    const int map_index = static_cast<int>(sequence_index % kRestirPtPairingMapCount);
    const int source_pixel = pairing_maps[map_index * pixel_count + pixel];
    if (source_pixel < 0 || source_pixel >= pixel_count || source_pixel == pixel ||
        !restir_pt_surfaces_compatible_gpu(surface, surfaces[source_pixel], surface.depth)) return;
    const GpuRestirPtReservoir source = restir_pt_unpack_gpu(input[source_pixel]);
    if (!restir_pt_valid_gpu(source)) return;
    GpuRestirPtSpatialState state;
    state.source_pixel = source_pixel;
    state.source_m = source.M;
    state.valid = 1;
    if (source.rc_length <= 2) {
        if (!restir_pt_validate_reconnection_gpu(*scene_ptr, settings, surface, source)) return;
        state.jacobian = restir_pt_reconnection_jacobian_gpu(surface, source);
        GpuRestirVisibilityRay visibility;
        if (!restir_pt_generate_connection_ray_gpu(surface, source,
                path_states[pixel].path_index, visibility)) return;
        rays[pixel] = visibility;
        visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
    }
    states[pixel] = state;
}

__global__ void restir_pt_paired_spatial_normalize_kernel(const GpuScene* scene_ptr,
    const GpuRestirSurface* surfaces, const GpuPackedRestirPtReservoir* canonical_reservoirs,
    const GpuPackedRestirPtReservoir* forward_shifted_reservoirs,
    const GpuPackedRestirPtReservoir* provisional_reservoirs,
    const GpuRestirPtSpatialState* states, const int* forward_visibility_results,
    GpuPackedRestirPtReservoir* output, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirPtReservoir canonical = restir_pt_unpack_gpu(canonical_reservoirs[pixel]);
    GpuRestirPtReservoir selected = restir_pt_unpack_gpu(provisional_reservoirs[pixel]);
    if (!restir_pt_valid_gpu(selected)) { output[pixel] = {}; return; }

    const GpuRestirPtSpatialState state = states[pixel];
    const float selected_target = restir_luminance_gpu(selected.target);
    float inverse_density = 0.0f;
    if (state.valid && state.source_pixel >= 0 && state.source_pixel < pixel_count &&
        state.source_m > 0) {
        if (state.selected_source != 0) {
            const GpuRestirPtReservoir source =
                restir_pt_unpack_gpu(canonical_reservoirs[state.source_pixel]);
            if (restir_pt_valid_gpu(source) && state.jacobian > 0.0f) {
                inverse_density = restir_luminance_gpu(source.target) / state.jacobian;
            }
        } else {
            const GpuRestirPtSpatialState reciprocal = states[state.source_pixel];
            if (reciprocal.valid && reciprocal.source_pixel == pixel &&
                reciprocal.jacobian > 0.0f &&
                forward_visibility_results[state.source_pixel] > 0) {
                Vec3 reciprocal_target;
                if (reciprocal.replayed != 0) {
                    reciprocal_target = restir_pt_unpack_gpu(
                        forward_shifted_reservoirs[state.source_pixel]).target;
                } else {
                    reciprocal_target = restir_pt_shift_target_gpu(*scene_ptr,
                        surfaces[state.source_pixel], surfaces[pixel], canonical);
                }
                inverse_density = restir_luminance_gpu(reciprocal_target) * reciprocal.jacobian;
            }
        }
    }
    if (!isfinite(inverse_density)) inverse_density = 0.0f;

    float pi = selected_target;
    float pi_sum = selected_target * static_cast<float>(canonical.M);
    if (state.valid && state.source_m > 0) {
        if (state.selected_source != 0) pi = inverse_density;
        pi_sum += inverse_density * static_cast<float>(state.source_m);
    }
    selected.weight_sum = selected_target > 0.0f && pi_sum > 0.0f
        ? selected.weight_sum * pi / (selected_target * pi_sum) : 0.0f;
    output[pixel] = restir_pt_pack_gpu(selected);
}

__global__ void restir_pt_spatial_select_kernel(const GpuScene* scene_ptr,
    const GpuRestirSurface* surfaces, const GpuPackedRestirPtReservoir* input,
    const GpuPackedRestirPtReservoir* shifted,
    GpuRestirPtSpatialState* states, const int* visibility_results,
    GpuPackedRestirPtReservoir* output, uint32_t sequence_index, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirSurface surface = surfaces[pixel];
    GpuRestirPtReservoir canonical = restir_pt_unpack_gpu(input[pixel]);
    if (!surface.valid) { output[pixel] = {}; return; }
    GpuRestirPtReservoir result;
    uint32_t rng = make_pixel_seed(static_cast<uint32_t>(pixel), sequence_index, 0x9b05688cu);
    bool selected_source = false;
    if (canonical.M > 0) restir_pt_combine_gpu(result, canonical, canonical.target,
        1.0f, canonical.M, rng_float(rng));
    GpuRestirPtSpatialState state = states[pixel];
    GpuRestirPtReservoir source;
    Vec3 source_target{};
    float source_jacobian = 0.0f;
    if (state.valid) {
        source = state.replayed != 0
            ? restir_pt_unpack_gpu(shifted[pixel])
            : restir_pt_unpack_gpu(input[state.source_pixel]);
        if (visibility_results[pixel] > 0) {
            source_target = state.replayed != 0 ? source.target :
                restir_pt_shift_target_gpu(*scene_ptr, surface,
                    surfaces[state.source_pixel], source);
        }
        source_jacobian = state.jacobian;
        selected_source = restir_pt_combine_gpu(result, source, source_target,
            source_jacobian, source.M, rng_float(rng));
    }
    const int total_m = result.M;
    result.M = total_m;
    if (selected_source && result.rc_length <= 2) {
        result.partial_jacobian = restir_pt_partial_inverse_jacobian_gpu(
            surface.position, result.position, result.normal);
    }
    state.selected_source = selected_source ? 1 : 0;
    states[pixel] = state;
    output[pixel] = restir_pt_pack_gpu(result);
}

__global__ void restir_pt_generate_final_visibility_kernel(
    const GpuRestirSurface* surfaces, const GpuPackedRestirPtReservoir* reservoirs,
    const GpuRestirPtPathState* states, GpuRestirVisibilityRay* rays, int* results,
    int* visibility_indices, int* visibility_count, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    rays[pixel] = {};
    results[pixel] = -1;
    const GpuRestirPtReservoir reservoir = restir_pt_unpack_gpu(reservoirs[pixel]);
    if (!surfaces[pixel].valid || !restir_pt_valid_gpu(reservoir) ||
        !(reservoir.weight_sum > 0.0f)) return;
    if (reservoir.rc_length > 2) {
        results[pixel] = 1;
        return;
    }
    GpuRestirVisibilityRay visibility;
    if (!restir_pt_generate_connection_ray_gpu(surfaces[pixel], reservoir,
            states[pixel].path_index, visibility)) return;
    rays[pixel] = visibility;
    visibility_indices[atomicAdd(visibility_count, 1)] = pixel;
}

__global__ void restir_pt_resolve_final_kernel(GpuWavefrontPaths paths,
    const GpuRestirPtPathState* states, const GpuPackedRestirPtReservoir* reservoirs,
    const int* visibility_results, Vec3* samples, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const GpuRestirPtPathState state = states[pixel];
    if (state.path_index < 0) return;
    GpuWavefrontPathRef path = wavefront_path_ref(paths, state.path_index);
    const GpuRestirPtReservoir reservoir = restir_pt_unpack_gpu(reservoirs[pixel]);
    Vec3 contribution{};
    if (visibility_results[pixel] > 0 && restir_pt_valid_gpu(reservoir)) {
        contribution = mul(reservoir.target, reservoir.weight_sum);
    }
    path.radiance = add(path.radiance,
        clamp_sample_radiance_gpu(mul(path.throughput, contribution), 8.0f));
    finish_wavefront_path(path, samples);
}

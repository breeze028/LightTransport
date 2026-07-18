constexpr char kLightmapCacheMagic[8] = {'L', 'T', 'L', 'M', 'A', 'P', '1', '\0'};
constexpr uint32_t kLightmapCacheVersion = 1;

struct LightmapCacheHeader {
    char magic[8] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t fingerprint = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t triangle_count = 0;
    uint32_t bake_samples = 0;
    uint32_t bake_bounces = 0;
    uint32_t padding = 0;
    uint32_t dilation = 0;
};

void set_lightmap_progress_phase(const RenderSettings& settings, LightmapBakePhase phase) {
    if (settings.lightmap_bake_progress) {
        settings.lightmap_bake_progress->phase.store(static_cast<int>(phase), std::memory_order_relaxed);
    }
}

void reset_lightmap_progress(const RenderSettings& settings, LightmapBakePhase phase) {
    if (!settings.lightmap_bake_progress) {
        return;
    }
    LightmapBakeProgress& progress = *settings.lightmap_bake_progress;
    progress.total_texels.store(0, std::memory_order_relaxed);
    progress.completed_texels.store(0, std::memory_order_relaxed);
    progress.total_rays.store(0, std::memory_order_relaxed);
    progress.traced_rays.store(0, std::memory_order_relaxed);
    progress.width.store(0, std::memory_order_relaxed);
    progress.height.store(0, std::memory_order_relaxed);
    progress.elapsed_ms.store(0.0, std::memory_order_relaxed);
    progress.phase.store(static_cast<int>(phase), std::memory_order_relaxed);
}

std::string lightmap_cache_path(const RenderSettings& settings) {
    if (settings.lightmap_cache_path[0] != '\0') {
        return settings.lightmap_cache_path;
    }
    if (settings.lightmap_cache_key[0] != '\0') {
        return std::string(settings.lightmap_cache_key) + ".lmap";
    }
    return {};
}

uint64_t lightmap_fingerprint(const RenderScene& render_scene, const Scene& scene, const RenderSettings& settings) {
    uint64_t hash = kFnvOffsetBasis;
    hash_string(hash, "lt_lightmap_cache_v1");
    hash_c_string(hash, settings.lightmap_cache_key);
    hash_value(hash, static_cast<int>(settings.sampling_mode));
    hash_value(hash, static_cast<int>(settings.mis_heuristic));
    hash_value(hash, std::max(1, settings.lightmap_resolution));
    hash_value(hash, std::max(0, settings.lightmap_padding));
    hash_value(hash, std::max(0, settings.lightmap_dilation));
    hash_value(hash, std::max(1, settings.lightmap_bake_samples));
    hash_value(hash, std::max(1, settings.lightmap_bake_bounces));
    hash_value(hash, settings.lightmap_principled_gi);
    hash_value(hash, std::max(0.0f, settings.emissive_intensity_scale));

    const uint64_t triangle_count = static_cast<uint64_t>(render_scene.triangles.size());
    hash_value(hash, triangle_count);
    for (const Triangle& triangle : render_scene.triangles) {
        hash_vec3(hash, triangle.v0);
        hash_vec3(hash, triangle.v1);
        hash_vec3(hash, triangle.v2);
        hash_vec3(hash, triangle.normal);
        hash_vec3(hash, triangle.n0);
        hash_vec3(hash, triangle.n1);
        hash_vec3(hash, triangle.n2);
        hash_value(hash, triangle.material);
        hash_value(hash, triangle.mesh);
    }
    const uint64_t sphere_count = static_cast<uint64_t>(render_scene.spheres.size());
    hash_value(hash, sphere_count);
    for (const RenderSphere& sphere : render_scene.spheres) {
        hash_vec3(hash, sphere.center);
        hash_value(hash, sphere.radius);
        hash_value(hash, sphere.material);
    }
    const uint64_t material_count = static_cast<uint64_t>(scene.materials.size());
    hash_value(hash, material_count);
    for (const std::shared_ptr<Material>& material : scene.materials) {
        hash_material(hash, material);
    }
    const uint64_t mesh_count = static_cast<uint64_t>(scene.meshes.size());
    hash_value(hash, mesh_count);
    for (const Mesh& mesh : scene.meshes) {
        hash_value(hash, mesh.material);
        hash_value(hash, mesh.light.enabled);
        hash_value(hash, mesh.light.double_sided);
        hash_vec3(hash, mesh.light.color);
        hash_value(hash, mesh.light.intensity);
    }
    const uint64_t directional_light_count = static_cast<uint64_t>(scene.directional_lights.size());
    hash_value(hash, directional_light_count);
    for (const DirectionalLight& light : scene.directional_lights) {
        hash_vec3(hash, light.direction);
        hash_vec3(hash, light.color);
        hash_value(hash, light.intensity);
    }
    hash_vec3(hash, scene.environment.color);
    hash_value(hash, scene.environment.strength);
    hash_value(hash, scene.environment.constant);
    hash_value(hash, static_cast<int>(scene.environment.mapping));
    hash_vec3(hash, scene.environment.light_from_world_x);
    hash_vec3(hash, scene.environment.light_from_world_y);
    hash_vec3(hash, scene.environment.light_from_world_z);
    hash_texture_ref(hash, scene.environment.texture);
    return hash;
}

std::shared_ptr<Lightmap> load_lightmap_cache(const RenderSettings& settings, uint64_t fingerprint) {
    const std::string path = lightmap_cache_path(settings);
    if (!settings.lightmap_cache_enabled || path.empty()) {
        return nullptr;
    }
    reset_lightmap_progress(settings, LightmapBakePhase::LoadingCache);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        set_lightmap_progress_phase(settings, LightmapBakePhase::Idle);
        return nullptr;
    }
    const auto begin = std::chrono::steady_clock::now();
    LightmapCacheHeader header;
    if (!read_binary(input, header) ||
        std::memcmp(header.magic, kLightmapCacheMagic, sizeof(header.magic)) != 0 ||
        header.version != kLightmapCacheVersion ||
        header.header_size != sizeof(LightmapCacheHeader) ||
        header.fingerprint != fingerprint ||
        header.width == 0 || header.height == 0 || header.width > 16384u || header.height > 16384u) {
        set_lightmap_progress_phase(settings, LightmapBakePhase::Idle);
        return nullptr;
    }
    auto lightmap = std::make_shared<Lightmap>();
    lightmap->width = static_cast<int>(header.width);
    lightmap->height = static_cast<int>(header.height);
    lightmap->bake_samples = static_cast<int>(header.bake_samples);
    lightmap->bake_bounces = static_cast<int>(header.bake_bounces);
    lightmap->triangles.resize(header.triangle_count);
    for (LightmapTriangle& triangle : lightmap->triangles) {
        uint8_t valid = 0;
        if (!read_binary(input, triangle.uv0) || !read_binary(input, triangle.uv1) ||
            !read_binary(input, triangle.uv2) || !read_binary(input, valid)) {
            set_lightmap_progress_phase(settings, LightmapBakePhase::Failed);
            return nullptr;
        }
        triangle.valid = valid != 0;
    }
    const size_t texel_count = static_cast<size_t>(lightmap->width) * static_cast<size_t>(lightmap->height);
    lightmap->texels.resize(texel_count);
    lightmap->valid.resize(texel_count);
    for (size_t i = 0; i < texel_count; ++i) {
        if (!read_binary(input, lightmap->texels[i]) || !read_binary(input, lightmap->valid[i])) {
            set_lightmap_progress_phase(settings, LightmapBakePhase::Failed);
            return nullptr;
        }
        if (lightmap->valid[i]) {
            ++lightmap->baked_texel_count;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    if (settings.lightmap_bake_progress) {
        settings.lightmap_bake_progress->total_texels.store(lightmap->baked_texel_count, std::memory_order_relaxed);
        settings.lightmap_bake_progress->completed_texels.store(lightmap->baked_texel_count, std::memory_order_relaxed);
        settings.lightmap_bake_progress->width.store(lightmap->width, std::memory_order_relaxed);
        settings.lightmap_bake_progress->height.store(lightmap->height, std::memory_order_relaxed);
        settings.lightmap_bake_progress->elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
        settings.lightmap_bake_progress->phase.store(static_cast<int>(LightmapBakePhase::Complete), std::memory_order_relaxed);
    }
    LT_LOG_INFO("Lightmap cache loaded: path='{}' size={}x{} texels={} elapsed_ms={}",
        path, lightmap->width, lightmap->height, lightmap->baked_texel_count, format_decimal(elapsed_ms, 3));
    return lightmap;
}

bool save_lightmap_cache(const Lightmap& lightmap, const RenderSettings& settings, uint64_t fingerprint) {
    const std::string path = lightmap_cache_path(settings);
    if (!settings.lightmap_cache_enabled || path.empty()) {
        return false;
    }
    set_lightmap_progress_phase(settings, LightmapBakePhase::SavingCache);
    std::error_code error;
    const std::filesystem::path cache_path(path);
    const std::filesystem::path parent = cache_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            LT_LOG_WARN("Could not create lightmap cache directory '{}': {}", parent.string(), error.message());
            set_lightmap_progress_phase(settings, LightmapBakePhase::Complete);
            return false;
        }
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        LT_LOG_WARN("Could not write lightmap cache '{}'", path);
        set_lightmap_progress_phase(settings, LightmapBakePhase::Complete);
        return false;
    }
    LightmapCacheHeader header;
    std::memcpy(header.magic, kLightmapCacheMagic, sizeof(header.magic));
    header.version = kLightmapCacheVersion;
    header.header_size = sizeof(LightmapCacheHeader);
    header.fingerprint = fingerprint;
    header.width = static_cast<uint32_t>(std::max(0, lightmap.width));
    header.height = static_cast<uint32_t>(std::max(0, lightmap.height));
    header.triangle_count = static_cast<uint32_t>(lightmap.triangles.size());
    header.bake_samples = static_cast<uint32_t>(std::max(1, lightmap.bake_samples));
    header.bake_bounces = static_cast<uint32_t>(std::max(1, lightmap.bake_bounces));
    header.padding = static_cast<uint32_t>(std::max(0, settings.lightmap_padding));
    header.dilation = static_cast<uint32_t>(std::max(0, settings.lightmap_dilation));
    if (!write_binary(output, header)) {
        set_lightmap_progress_phase(settings, LightmapBakePhase::Complete);
        return false;
    }
    for (const LightmapTriangle& triangle : lightmap.triangles) {
        const uint8_t valid = triangle.valid ? 1u : 0u;
        if (!write_binary(output, triangle.uv0) || !write_binary(output, triangle.uv1) ||
            !write_binary(output, triangle.uv2) || !write_binary(output, valid)) {
            set_lightmap_progress_phase(settings, LightmapBakePhase::Complete);
            return false;
        }
    }
    for (size_t i = 0; i < lightmap.texels.size(); ++i) {
        const uint8_t valid = i < lightmap.valid.size() ? lightmap.valid[i] : 0u;
        if (!write_binary(output, lightmap.texels[i]) || !write_binary(output, valid)) {
            set_lightmap_progress_phase(settings, LightmapBakePhase::Complete);
            return false;
        }
    }
    LT_LOG_INFO("Lightmap cache saved: path='{}'", path);
    set_lightmap_progress_phase(settings, LightmapBakePhase::Complete);
    return true;
}

uint8_t lightmap_preview_channel(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    value = std::pow(value, 1.0f / 2.2f);
    return static_cast<uint8_t>(std::clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
}

std::string lightmap_dds_path(const RenderSettings& settings) {
    std::string path = lightmap_cache_path(settings);
    if (path.empty()) {
        return {};
    }
    const std::filesystem::path cache_path(path);
    std::filesystem::path dds_path(path);
    dds_path.replace_extension(".dds");
    if (dds_path == cache_path) {
        dds_path.replace_extension(".preview.dds");
    }
    return dds_path.string();
}

bool save_lightmap_dds(const Lightmap& lightmap, const RenderSettings& settings) {
    const std::string path = lightmap_dds_path(settings);
    if (path.empty() || lightmap.width <= 0 || lightmap.height <= 0 || lightmap.texels.empty()) {
        return false;
    }
    std::error_code error;
    const std::filesystem::path output_path(path);
    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            LT_LOG_WARN("Could not create lightmap DDS directory '{}': {}", parent.string(), error.message());
            return false;
        }
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        LT_LOG_WARN("Could not write lightmap DDS '{}'", path);
        return false;
    }

    const uint32_t magic = 0x20534444u;
    const uint32_t header_size = 124u;
    const uint32_t flags = 0x00001007u;
    const uint32_t height = static_cast<uint32_t>(lightmap.height);
    const uint32_t width = static_cast<uint32_t>(lightmap.width);
    const uint32_t pitch = width * 4u;
    const uint32_t zero = 0u;
    const uint32_t pixel_format_size = 32u;
    const uint32_t pixel_format_flags = 0x00000041u;
    const uint32_t rgb_bit_count = 32u;
    const uint32_t r_mask = 0x00ff0000u;
    const uint32_t g_mask = 0x0000ff00u;
    const uint32_t b_mask = 0x000000ffu;
    const uint32_t a_mask = 0xff000000u;
    const uint32_t caps = 0x00001000u;

    write_binary(output, magic);
    write_binary(output, header_size);
    write_binary(output, flags);
    write_binary(output, height);
    write_binary(output, width);
    write_binary(output, pitch);
    write_binary(output, zero);
    write_binary(output, zero);
    for (int i = 0; i < 11; ++i) write_binary(output, zero);
    write_binary(output, pixel_format_size);
    write_binary(output, pixel_format_flags);
    write_binary(output, zero);
    write_binary(output, rgb_bit_count);
    write_binary(output, r_mask);
    write_binary(output, g_mask);
    write_binary(output, b_mask);
    write_binary(output, a_mask);
    write_binary(output, caps);
    write_binary(output, zero);
    write_binary(output, zero);
    write_binary(output, zero);
    write_binary(output, zero);

    for (size_t i = 0; i < lightmap.texels.size(); ++i) {
        const Vec3 value = lightmap.texels[i];
        const uint8_t rgba[4] = {
            lightmap_preview_channel(value.z),
            lightmap_preview_channel(value.y),
            lightmap_preview_channel(value.x),
            i < lightmap.valid.size() && lightmap.valid[i] ? uint8_t{255} : uint8_t{0},
        };
        output.write(reinterpret_cast<const char*>(rgba), sizeof(rgba));
    }
    if (!output) {
        LT_LOG_WARN("Could not write complete lightmap DDS '{}'", path);
        return false;
    }
    LT_LOG_INFO("Lightmap DDS saved: path='{}'", path);
    return true;
}

bool save_lightmap_outputs(const Lightmap& lightmap, const RenderSettings& settings, uint64_t fingerprint) {
    const bool cache_saved = save_lightmap_cache(lightmap, settings, fingerprint);
    save_lightmap_dds(lightmap, settings);
    return cache_saved;
}

bool generate_lightmap_uvs(const RenderScene& render_scene, const RenderSettings& settings, Lightmap& lightmap) {
    reset_lightmap_progress(settings, LightmapBakePhase::Unwrapping);
    lightmap.triangles.assign(render_scene.triangles.size(), {});
    if (render_scene.triangles.empty()) {
        return false;
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<uint32_t> indices;
    positions.reserve(render_scene.triangles.size() * 3u);
    normals.reserve(render_scene.triangles.size() * 3u);
    indices.reserve(render_scene.triangles.size() * 3u);
    for (const Triangle& triangle : render_scene.triangles) {
        const uint32_t base = static_cast<uint32_t>(positions.size());
        positions.push_back(triangle.v0);
        positions.push_back(triangle.v1);
        positions.push_back(triangle.v2);
        normals.push_back(triangle.n0);
        normals.push_back(triangle.n1);
        normals.push_back(triangle.n2);
        indices.push_back(base);
        indices.push_back(base + 1u);
        indices.push_back(base + 2u);
    }

    ::xatlas::Atlas* atlas = ::xatlas::Create();
    ::xatlas::MeshDecl mesh_decl;
    mesh_decl.vertexPositionData = positions.data();
    mesh_decl.vertexPositionStride = sizeof(Vec3);
    mesh_decl.vertexNormalData = normals.data();
    mesh_decl.vertexNormalStride = sizeof(Vec3);
    mesh_decl.vertexCount = static_cast<uint32_t>(positions.size());
    mesh_decl.indexData = indices.data();
    mesh_decl.indexCount = static_cast<uint32_t>(indices.size());
    mesh_decl.indexFormat = ::xatlas::IndexFormat::UInt32;
    const ::xatlas::AddMeshError add_error = ::xatlas::AddMesh(atlas, mesh_decl);
    if (add_error != ::xatlas::AddMeshError::Success) {
        LT_LOG_WARN("Could not add mesh to xatlas: {}", ::xatlas::StringForEnum(add_error));
        ::xatlas::Destroy(atlas);
        set_lightmap_progress_phase(settings, LightmapBakePhase::Failed);
        return false;
    }
    ::xatlas::ChartOptions chart_options;
    ::xatlas::PackOptions pack_options;
    pack_options.resolution = static_cast<uint32_t>(std::clamp(settings.lightmap_resolution, 16, 16384));
    pack_options.padding = static_cast<uint32_t>(std::clamp(settings.lightmap_padding, 0, 64));
    pack_options.bilinear = true;
    pack_options.createImage = false;
    ::xatlas::Generate(atlas, chart_options, pack_options);
    if (atlas->meshCount != 1 || !atlas->meshes || atlas->width == 0 || atlas->height == 0) {
        ::xatlas::Destroy(atlas);
        set_lightmap_progress_phase(settings, LightmapBakePhase::Failed);
        return false;
    }
    lightmap.width = static_cast<int>(atlas->width);
    lightmap.height = static_cast<int>(atlas->height);
    lightmap.bake_samples = std::max(1, settings.lightmap_bake_samples);
    lightmap.bake_bounces = std::max(1, settings.lightmap_bake_bounces);
    lightmap.texels.assign(static_cast<size_t>(lightmap.width) * static_cast<size_t>(lightmap.height), {});
    lightmap.valid.assign(lightmap.texels.size(), 0u);

    const ::xatlas::Mesh& mesh = atlas->meshes[0];
    const uint32_t face_count = mesh.indexCount / 3u;
    size_t mapped_triangle_count = 0;
    for (uint32_t face = 0; face < face_count && face < lightmap.triangles.size(); ++face) {
        const ::xatlas::Vertex& v0 = mesh.vertexArray[mesh.indexArray[face * 3u + 0u]];
        const ::xatlas::Vertex& v1 = mesh.vertexArray[mesh.indexArray[face * 3u + 1u]];
        const ::xatlas::Vertex& v2 = mesh.vertexArray[mesh.indexArray[face * 3u + 2u]];
        if (v0.atlasIndex < 0 || v1.atlasIndex < 0 || v2.atlasIndex < 0) {
            continue;
        }
        LightmapTriangle& triangle = lightmap.triangles[face];
        triangle.uv0 = {v0.uv[0] / static_cast<float>(lightmap.width), v0.uv[1] / static_cast<float>(lightmap.height)};
        triangle.uv1 = {v1.uv[0] / static_cast<float>(lightmap.width), v1.uv[1] / static_cast<float>(lightmap.height)};
        triangle.uv2 = {v2.uv[0] / static_cast<float>(lightmap.width), v2.uv[1] / static_cast<float>(lightmap.height)};
        triangle.valid = true;
        ++mapped_triangle_count;
    }
    LT_LOG_INFO("Lightmap UVs generated: triangles={} mapped={} atlas={}x{}",
        render_scene.triangles.size(), mapped_triangle_count, lightmap.width, lightmap.height);
    ::xatlas::Destroy(atlas);
    return true;
}

float edge_function(Vec2 a, Vec2 b, Vec2 p) {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

float luminance(Vec3 value) {
    return value.x * 0.2126f + value.y * 0.7152f + value.z * 0.0722f;
}

Vec3 cosine_hemisphere_world(Vec3 normal, float u, float v) {
    const Vec2 disk = concentric_sample_disk(u, v);
    const float z = std::sqrt(std::max(0.0f, 1.0f - disk.x * disk.x - disk.y * disk.y));
    const Vec3 tangent = normalize(cross(std::fabs(normal.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f}, normal));
    const Vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * disk.x + bitangent * disk.y + normal * z);
}

Vec3 trace_lightmap_radiance(
    const RenderScene& render_scene,
    const Scene& scene,
    Vec3 origin,
    Vec3 normal,
    Rng& rng,
    RenderSettings bake_settings,
    size_t& traced_rays) {
    bake_settings.use_lightmap = false;
    bake_settings.use_irradiance_volume = false;
    bake_settings.stylized_samples = 0;
    bake_settings.stylized_max_depth = 0;
    bake_settings.max_bounces = std::max(1, bake_settings.lightmap_bake_bounces);

    const Vec3 direction = cosine_hemisphere_world(normal, rng.next_float(), rng.next_float());
    Ray ray{origin + normal * 0.002f, direction};
    ++traced_rays;
    Rng visibility_rng = rng;
    for (int step = 0; step < 8; ++step) {
        Hit first_hit;
        if (!intersect_scene(render_scene, ray, first_hit, bake_settings.acceleration_structure)) {
            return {};
        }
        if (first_hit.material < 0 || first_hit.material >= static_cast<int>(scene.materials.size())) {
            return {};
        }
        const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(first_hit.material)];
        if (!material) {
            return {};
        }
        if (!material_visible(*material, first_hit.uv, visibility_rng)) {
            ray = {first_hit.position + direction * 0.002f, direction};
            continue;
        }
        Vec3 emission;
        if (first_hit.triangle >= 0 && first_hit.triangle < static_cast<int>(render_scene.triangles.size())) {
            emission = emitted_radiance(scene, render_scene.triangles[static_cast<size_t>(first_hit.triangle)], first_hit.uv, direction, bake_settings);
        }
        if (has_light_emission(emission)) {
            return {};
        }
        break;
    }
    return trace_path(render_scene, scene, ray, rng, bake_settings, nullptr, nullptr) * kPi;
}

void bake_lightmap_texels(Lightmap& lightmap, const RenderScene& render_scene, const Scene& scene, const RenderSettings& settings) {
    reset_lightmap_progress(settings, LightmapBakePhase::Baking);
    std::vector<int> owner(lightmap.texels.size(), -1);
    for (size_t triangle_index = 0; triangle_index < render_scene.triangles.size() && triangle_index < lightmap.triangles.size(); ++triangle_index) {
        const LightmapTriangle& lm_tri = lightmap.triangles[triangle_index];
        if (!lm_tri.valid) {
            continue;
        }
        const Vec2 p0{lm_tri.uv0.x * static_cast<float>(lightmap.width), lm_tri.uv0.y * static_cast<float>(lightmap.height)};
        const Vec2 p1{lm_tri.uv1.x * static_cast<float>(lightmap.width), lm_tri.uv1.y * static_cast<float>(lightmap.height)};
        const Vec2 p2{lm_tri.uv2.x * static_cast<float>(lightmap.width), lm_tri.uv2.y * static_cast<float>(lightmap.height)};
        const float area = edge_function(p0, p1, p2);
        if (std::fabs(area) <= 1.0e-8f) {
            continue;
        }
        const int min_x = std::clamp(static_cast<int>(std::floor(std::min({p0.x, p1.x, p2.x}))) - 1, 0, lightmap.width - 1);
        const int max_x = std::clamp(static_cast<int>(std::ceil(std::max({p0.x, p1.x, p2.x}))) + 1, 0, lightmap.width - 1);
        const int min_y = std::clamp(static_cast<int>(std::floor(std::min({p0.y, p1.y, p2.y}))) - 1, 0, lightmap.height - 1);
        const int max_y = std::clamp(static_cast<int>(std::ceil(std::max({p0.y, p1.y, p2.y}))) + 1, 0, lightmap.height - 1);
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                const Vec2 p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
                const float w0 = edge_function(p1, p2, p) / area;
                const float w1 = edge_function(p2, p0, p) / area;
                const float w2 = edge_function(p0, p1, p) / area;
                if (w0 >= -1.0e-4f && w1 >= -1.0e-4f && w2 >= -1.0e-4f) {
                    owner[static_cast<size_t>(lightmap_texel_index(lightmap, x, y))] = static_cast<int>(triangle_index);
                }
            }
        }
    }
    size_t total_texels = 0;
    for (int value : owner) {
        if (value >= 0) {
            ++total_texels;
        }
    }
    if (settings.lightmap_bake_progress) {
        settings.lightmap_bake_progress->total_texels.store(total_texels, std::memory_order_relaxed);
        settings.lightmap_bake_progress->total_rays.store(total_texels * static_cast<size_t>(std::max(1, lightmap.bake_samples)) * 2u, std::memory_order_relaxed);
        settings.lightmap_bake_progress->width.store(lightmap.width, std::memory_order_relaxed);
        settings.lightmap_bake_progress->height.store(lightmap.height, std::memory_order_relaxed);
        settings.lightmap_bake_progress->phase.store(static_cast<int>(LightmapBakePhase::Baking), std::memory_order_relaxed);
    }

    for (size_t texel = 0; texel < owner.size(); ++texel) {
        const int triangle_index = owner[texel];
        if (triangle_index < 0) {
            continue;
        }
        const int x = static_cast<int>(texel % static_cast<size_t>(lightmap.width));
        const int y = static_cast<int>(texel / static_cast<size_t>(lightmap.width));
        const LightmapTriangle& lm_tri = lightmap.triangles[static_cast<size_t>(triangle_index)];
        const Triangle& tri = render_scene.triangles[static_cast<size_t>(triangle_index)];
        const Vec2 p0{lm_tri.uv0.x * static_cast<float>(lightmap.width), lm_tri.uv0.y * static_cast<float>(lightmap.height)};
        const Vec2 p1{lm_tri.uv1.x * static_cast<float>(lightmap.width), lm_tri.uv1.y * static_cast<float>(lightmap.height)};
        const Vec2 p2{lm_tri.uv2.x * static_cast<float>(lightmap.width), lm_tri.uv2.y * static_cast<float>(lightmap.height)};
        const float area = edge_function(p0, p1, p2);
        const Vec2 p{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
        const float w0 = edge_function(p1, p2, p) / area;
        const float w1 = edge_function(p2, p0, p) / area;
        const float w2 = 1.0f - w0 - w1;
        const Vec3 position = tri.v0 * w0 + tri.v1 * w1 + tri.v2 * w2;
        Vec3 normal = normalize(tri.n0 * w0 + tri.n1 * w1 + tri.n2 * w2);
        if (dot(normal, normal) <= 0.0f) {
            normal = tri.normal;
        }
        Vec3 front_irradiance;
        Vec3 back_irradiance;
        size_t traced_rays = 0;
        for (int sample_index = 0; sample_index < lightmap.bake_samples; ++sample_index) {
            const uint32_t seed = hash_u32(
                static_cast<uint32_t>(texel) * 0x8da6b343u ^
                static_cast<uint32_t>(sample_index) * 0xcb1ab31fu ^
                0x51ed270bu);
            Rng rng(seed);
            front_irradiance += trace_lightmap_radiance(render_scene, scene, position, normal, rng, settings, traced_rays);
            Rng back_rng(hash_u32(seed ^ 0xa511e9b3u));
            back_irradiance += trace_lightmap_radiance(render_scene, scene, position, -normal, back_rng, settings, traced_rays);
        }
        front_irradiance = front_irradiance / static_cast<float>(lightmap.bake_samples);
        back_irradiance = back_irradiance / static_cast<float>(lightmap.bake_samples);
        const Vec3 irradiance = luminance(back_irradiance) > luminance(front_irradiance) ? back_irradiance : front_irradiance;
        lightmap.texels[texel] = clamp_sample_radiance(irradiance, 1024.0f);
        lightmap.valid[texel] = 1u;
        ++lightmap.baked_texel_count;
        lightmap.traced_ray_count += traced_rays;
        if (settings.lightmap_bake_progress) {
            settings.lightmap_bake_progress->completed_texels.fetch_add(1, std::memory_order_relaxed);
            settings.lightmap_bake_progress->traced_rays.fetch_add(traced_rays, std::memory_order_relaxed);
        }
    }
}

void dilate_lightmap(Lightmap& lightmap, int iterations) {
    iterations = std::clamp(iterations, 0, 64);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<Vec3> next_texels = lightmap.texels;
        std::vector<uint8_t> next_valid = lightmap.valid;
        for (int y = 0; y < lightmap.height; ++y) {
            for (int x = 0; x < lightmap.width; ++x) {
                const int index = lightmap_texel_index(lightmap, x, y);
                if (lightmap.valid[static_cast<size_t>(index)]) {
                    continue;
                }
                Vec3 sum;
                int count = 0;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        if (ox == 0 && oy == 0) {
                            continue;
                        }
                        const int nx = x + ox;
                        const int ny = y + oy;
                        if (nx < 0 || nx >= lightmap.width || ny < 0 || ny >= lightmap.height) {
                            continue;
                        }
                        const int neighbor = lightmap_texel_index(lightmap, nx, ny);
                        if (lightmap.valid[static_cast<size_t>(neighbor)]) {
                            sum += lightmap.texels[static_cast<size_t>(neighbor)];
                            ++count;
                        }
                    }
                }
                if (count > 0) {
                    next_texels[static_cast<size_t>(index)] = sum / static_cast<float>(count);
                    next_valid[static_cast<size_t>(index)] = 1u;
                }
            }
        }
        lightmap.texels = std::move(next_texels);
        lightmap.valid = std::move(next_valid);
    }
}

std::shared_ptr<Lightmap> build_lightmap(const RenderScene& render_scene, const Scene& scene, const RenderSettings& settings) {
    const auto begin = std::chrono::steady_clock::now();
    auto lightmap = std::make_shared<Lightmap>();
    if (!generate_lightmap_uvs(render_scene, settings, *lightmap)) {
        return nullptr;
    }
    bake_lightmap_texels(*lightmap, render_scene, scene, settings);
    dilate_lightmap(*lightmap, settings.lightmap_dilation);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    if (settings.lightmap_bake_progress) {
        settings.lightmap_bake_progress->elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
        settings.lightmap_bake_progress->phase.store(static_cast<int>(LightmapBakePhase::Complete), std::memory_order_relaxed);
    }
    LT_LOG_INFO("Lightmap baked: size={}x{} texels={} rays={} memory_kib={} elapsed_ms={} elapsed_s={}",
        lightmap->width,
        lightmap->height,
        lightmap->baked_texel_count,
        lightmap->traced_ray_count,
        (lightmap->texels.size() * sizeof(Vec3)) / 1024u,
        format_decimal(elapsed_ms, 3),
        format_decimal(elapsed_ms * 0.001, 2));
    return lightmap;
}

std::shared_ptr<Lightmap> update_lightmap(
    std::shared_ptr<void>& cached_lightmap,
    RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings,
    bool lightmap_dirty,
    bool& lightmap_rebuilt) {
    lightmap_rebuilt = false;
    const bool force_rebake = settings.lightmap_force_rebake;
    const bool needs_update = !cached_lightmap || lightmap_dirty || force_rebake;
    if (!needs_update) {
        set_lightmap_progress_phase(settings, LightmapBakePhase::Idle);
        std::shared_ptr<Lightmap> lightmap = std::static_pointer_cast<Lightmap>(cached_lightmap);
        apply_lightmap_to_render_scene(lightmap.get(), render_scene);
        return lightmap;
    }
    const uint64_t fingerprint = lightmap_fingerprint(render_scene, scene, settings);
    if (!force_rebake) {
        if (std::shared_ptr<Lightmap> cached = load_lightmap_cache(settings, fingerprint)) {
            cached_lightmap = cached;
            lightmap_rebuilt = true;
            apply_lightmap_to_render_scene(cached.get(), render_scene);
            return cached;
        }
    }
    if (cached_lightmap && !settings.lightmap_auto_update && !force_rebake) {
        LT_LOG_INFO("Lightmap auto update disabled; keeping existing in-memory lightmap");
        set_lightmap_progress_phase(settings, LightmapBakePhase::Idle);
        std::shared_ptr<Lightmap> lightmap = std::static_pointer_cast<Lightmap>(cached_lightmap);
        apply_lightmap_to_render_scene(lightmap.get(), render_scene);
        return lightmap;
    }
    if (!settings.lightmap_auto_update && !force_rebake) {
        LT_LOG_WARN("Lightmap auto update disabled, but no usable cached lightmap exists; baking initial lightmap");
    }

    std::shared_ptr<Lightmap> lightmap;
    if (settings.lightmap_bake_backend == LightmapBakeBackend::Gpu) {
        std::shared_ptr<void> result = build_lightmap_gpu(render_scene, scene, settings);
        lightmap = std::static_pointer_cast<Lightmap>(result);
    } else {
        lightmap = build_lightmap(render_scene, scene, settings);
    }

    if (!lightmap) {
        if (cached_lightmap) {
            LT_LOG_WARN("Lightmap bake failed with {} backend; keeping previous lightmap",
                settings.lightmap_bake_backend == LightmapBakeBackend::Gpu ? "GPU" : "CPU");
            set_lightmap_progress_phase(settings, LightmapBakePhase::Failed);
            std::shared_ptr<Lightmap> prev = std::static_pointer_cast<Lightmap>(cached_lightmap);
            apply_lightmap_to_render_scene(prev.get(), render_scene);
            return prev;
        }
        cached_lightmap.reset();
        apply_lightmap_to_render_scene(nullptr, render_scene);
        lightmap_rebuilt = true;
        set_lightmap_progress_phase(settings, LightmapBakePhase::Failed);
        return nullptr;
    }
    cached_lightmap = lightmap;
    lightmap_rebuilt = true;
    apply_lightmap_to_render_scene(lightmap.get(), render_scene);
    save_lightmap_outputs(*lightmap, settings, fingerprint);
    return lightmap;
}

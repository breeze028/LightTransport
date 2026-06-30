void expand_irradiance_bounds(Aabb& bounds, Vec3 point) {
    bounds.min = min(bounds.min, point);
    bounds.max = max(bounds.max, point);
}

Aabb triangle_irradiance_bounds(const Triangle& triangle) {
    Aabb bounds;
    expand_irradiance_bounds(bounds, triangle.v0);
    expand_irradiance_bounds(bounds, triangle.v1);
    expand_irradiance_bounds(bounds, triangle.v2);
    return bounds;
}

Aabb sphere_irradiance_bounds(const RenderSphere& sphere) {
    const Vec3 r{sphere.radius};
    return {sphere.center - r, sphere.center + r};
}

Aabb scene_irradiance_bounds(const RenderScene& render_scene, const RenderSettings& settings) {
    Aabb bounds;
    if (settings.irradiance_volume_manual_bounds) {
        bounds.min = min(settings.irradiance_volume_bounds_min, settings.irradiance_volume_bounds_max);
        bounds.max = max(settings.irradiance_volume_bounds_min, settings.irradiance_volume_bounds_max);
    } else {
        for (const Triangle& triangle : render_scene.triangles) {
            expand_irradiance_bounds(bounds, triangle.v0);
            expand_irradiance_bounds(bounds, triangle.v1);
            expand_irradiance_bounds(bounds, triangle.v2);
        }
        for (const RenderSphere& sphere : render_scene.spheres) {
            const Vec3 r{sphere.radius};
            expand_irradiance_bounds(bounds, sphere.center - r);
            expand_irradiance_bounds(bounds, sphere.center + r);
        }
    }

    if (!aabb_is_valid(bounds)) {
        bounds.min = {-1.0f, -1.0f, -1.0f};
        bounds.max = {1.0f, 1.0f, 1.0f};
        return bounds;
    }

    const Vec3 extent = bounds.max - bounds.min;
    const float max_extent = std::max({extent.x, extent.y, extent.z, 1.0e-3f});
    for (int axis = 0; axis < 3; ++axis) {
        if (axis_component(extent, axis) <= 1.0e-5f) {
            const float center = (axis_component(bounds.min, axis) + axis_component(bounds.max, axis)) * 0.5f;
            bounds.min = with_axis(bounds.min, axis, center - max_extent * 0.5f);
            bounds.max = with_axis(bounds.max, axis, center + max_extent * 0.5f);
        }
    }

    const float inset = std::clamp(settings.irradiance_volume_bounds_inset, 0.0f, 0.45f);
    if (inset > 0.0f) {
        const Aabb original = bounds;
        const Vec3 inset_amount = (bounds.max - bounds.min) * inset;
        bounds.min += inset_amount;
        bounds.max = bounds.max - inset_amount;
        if (!aabb_is_valid(bounds)) {
            bounds = original;
        }
    }
    return bounds;
}

bool nearly_equal_float(float a, float b, float epsilon) {
    return std::fabs(a - b) <= epsilon;
}

bool nearly_equal_vec3(Vec3 a, Vec3 b, float epsilon) {
    return nearly_equal_float(a.x, b.x, epsilon) &&
        nearly_equal_float(a.y, b.y, epsilon) &&
        nearly_equal_float(a.z, b.z, epsilon);
}

bool nearly_equal_bounds(const Aabb& a, const Aabb& b) {
    const Vec3 extent_a = a.max - a.min;
    const Vec3 extent_b = b.max - b.min;
    const float scale = std::max({
        std::fabs(extent_a.x), std::fabs(extent_a.y), std::fabs(extent_a.z),
        std::fabs(extent_b.x), std::fabs(extent_b.y), std::fabs(extent_b.z),
        1.0f,
    });
    const float epsilon = scale * 1.0e-4f;
    return nearly_equal_vec3(a.min, b.min, epsilon) && nearly_equal_vec3(a.max, b.max, epsilon);
}

bool cell_contains_irradiance_geometry(const RenderScene& render_scene, const Aabb& cell_bounds) {
    for (const Triangle& triangle : render_scene.triangles) {
        if (aabb_overlaps(cell_bounds, triangle_irradiance_bounds(triangle))) {
            return true;
        }
    }
    for (const RenderSphere& sphere : render_scene.spheres) {
        if (aabb_overlaps(cell_bounds, sphere_irradiance_bounds(sphere))) {
            return true;
        }
    }
    return false;
}

constexpr char kIrradianceVolumeCacheMagic[8] = {'L', 'T', 'I', 'V', 'O', 'L', '2', '\0'};
constexpr uint32_t kIrradianceVolumeCacheVersion = 1;
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

struct IrradianceVolumeCacheHeader {
    char magic[8] = {};
    uint32_t version = 0;
    uint32_t header_size = 0;
    uint64_t fingerprint = 0;
    uint32_t grid_resolution = 0;
    uint32_t subgrid_resolution = 0;
    uint32_t direction_resolution = 0;
    uint32_t bake_samples = 0;
    uint32_t bake_bounces = 0;
    Vec3 bounds_min;
    Vec3 bounds_max;
};

void set_irradiance_volume_progress_phase(const RenderSettings& settings, IrradianceVolumeBakePhase phase) {
    if (settings.irradiance_volume_bake_progress) {
        settings.irradiance_volume_bake_progress->phase.store(static_cast<int>(phase), std::memory_order_relaxed);
    }
}

void reset_irradiance_volume_progress(const RenderSettings& settings, IrradianceVolumeBakePhase phase) {
    if (!settings.irradiance_volume_bake_progress) {
        return;
    }
    IrradianceVolumeBakeProgress& progress = *settings.irradiance_volume_bake_progress;
    progress.total_samples.store(0, std::memory_order_relaxed);
    progress.completed_samples.store(0, std::memory_order_relaxed);
    progress.total_rays.store(0, std::memory_order_relaxed);
    progress.traced_rays.store(0, std::memory_order_relaxed);
    progress.direction_count.store(0, std::memory_order_relaxed);
    progress.elapsed_ms.store(0.0, std::memory_order_relaxed);
    progress.phase.store(static_cast<int>(phase), std::memory_order_relaxed);
}

std::string format_decimal(double value, int precision) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    return output.str();
}

void hash_bytes(uint64_t& hash, const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kFnvPrime;
    }
}

template <typename T>
void hash_value(uint64_t& hash, const T& value) {
    hash_bytes(hash, &value, sizeof(T));
}

void hash_string(uint64_t& hash, const std::string& value) {
    const uint64_t size = static_cast<uint64_t>(value.size());
    hash_value(hash, size);
    if (!value.empty()) {
        hash_bytes(hash, value.data(), value.size());
    }
}

void hash_c_string(uint64_t& hash, const char* value) {
    hash_string(hash, value ? std::string(value) : std::string{});
}

void hash_vec2(uint64_t& hash, Vec2 value) {
    hash_value(hash, value.x);
    hash_value(hash, value.y);
}

void hash_vec3(uint64_t& hash, Vec3 value) {
    hash_value(hash, value.x);
    hash_value(hash, value.y);
    hash_value(hash, value.z);
}

void hash_texture_ref(uint64_t& hash, const std::shared_ptr<Texture>& texture) {
    const bool has_texture = static_cast<bool>(texture);
    hash_value(hash, has_texture);
    if (!texture) {
        return;
    }
    hash_string(hash, texture->name);
    hash_string(hash, texture->path);
    hash_value(hash, texture->width);
    hash_value(hash, texture->height);
    const uint64_t pixel_count = static_cast<uint64_t>(texture->pixels.size());
    const uint64_t alpha_count = static_cast<uint64_t>(texture->alpha.size());
    hash_value(hash, pixel_count);
    hash_value(hash, alpha_count);
    const size_t pixel_stride = texture->pixels.empty() ? 1u : std::max<size_t>(1u, texture->pixels.size() / 64u);
    for (size_t i = 0; i < texture->pixels.size(); i += pixel_stride) {
        hash_vec3(hash, texture->pixels[i]);
    }
    if (!texture->pixels.empty()) {
        hash_vec3(hash, texture->pixels.back());
    }
    const size_t alpha_stride = texture->alpha.empty() ? 1u : std::max<size_t>(1u, texture->alpha.size() / 64u);
    for (size_t i = 0; i < texture->alpha.size(); i += alpha_stride) {
        hash_value(hash, texture->alpha[i]);
    }
    if (!texture->alpha.empty()) {
        hash_value(hash, texture->alpha.back());
    }
}

void hash_material(uint64_t& hash, const std::shared_ptr<Material>& material) {
    const bool has_material = static_cast<bool>(material);
    hash_value(hash, has_material);
    if (!material) {
        return;
    }
    const int model = static_cast<int>(material->model());
    const int alpha_mode = static_cast<int>(material->alpha_mode);
    hash_string(hash, material->name);
    hash_value(hash, model);
    hash_vec3(hash, material->albedo);
    hash_value(hash, material->alpha);
    hash_value(hash, material->alpha_cutoff);
    hash_value(hash, alpha_mode);
    hash_value(hash, material->double_sided);
    hash_value(hash, material->normal_scale);
    hash_vec3(hash, material->emission);
    hash_texture_ref(hash, material->albedo_texture);
    hash_texture_ref(hash, material->normal_texture);
    hash_texture_ref(hash, material->emission_texture);
    if (const auto* principled = dynamic_cast<const PrincipledMaterial*>(material.get())) {
        hash_value(hash, principled->roughness);
        hash_value(hash, principled->metallic);
        hash_vec3(hash, principled->sheen_color);
        hash_value(hash, principled->sheen_roughness);
        hash_value(hash, principled->clearcoat);
        hash_value(hash, principled->clearcoat_roughness);
        hash_texture_ref(hash, principled->metallic_roughness_texture);
        hash_texture_ref(hash, principled->sheen_color_texture);
        hash_texture_ref(hash, principled->sheen_roughness_texture);
        hash_texture_ref(hash, principled->clearcoat_texture);
        hash_texture_ref(hash, principled->clearcoat_roughness_texture);
    } else if (const auto* dielectric = dynamic_cast<const DielectricMaterial*>(material.get())) {
        hash_value(hash, dielectric->ior);
        hash_vec3(hash, dielectric->transmission_tint);
    } else if (const auto* standard = dynamic_cast<const StandardSurfaceMaterial*>(material.get())) {
        hash_value(hash, standard->roughness);
        hash_value(hash, standard->metalness);
        hash_value(hash, standard->specular_weight);
        hash_value(hash, standard->specular_ior);
        hash_value(hash, standard->transmission_weight);
        hash_vec3(hash, standard->transmission_color);
        hash_value(hash, standard->coat_weight);
        hash_value(hash, standard->coat_roughness);
        hash_vec3(hash, standard->sheen_color);
        hash_value(hash, standard->sheen_weight);
        hash_value(hash, standard->sheen_roughness);
        hash_texture_ref(hash, standard->base_color_input.texture);
        hash_texture_ref(hash, standard->roughness_input.texture);
        hash_texture_ref(hash, standard->metalness_input.texture);
        hash_texture_ref(hash, standard->specular_weight_input.texture);
        hash_texture_ref(hash, standard->transmission_input.texture);
        hash_texture_ref(hash, standard->coat_input.texture);
        hash_texture_ref(hash, standard->coat_roughness_input.texture);
        hash_texture_ref(hash, standard->sheen_color_input.texture);
        hash_texture_ref(hash, standard->sheen_roughness_input.texture);
    }
}

uint64_t irradiance_volume_fingerprint(const RenderScene& render_scene, const Scene& scene, const RenderSettings& settings) {
    uint64_t hash = kFnvOffsetBasis;
    hash_string(hash, "lt_irradiance_volume_cache_v2");
    hash_c_string(hash, settings.irradiance_volume_cache_key);
    hash_value(hash, settings.use_mis);
    hash_value(hash, static_cast<int>(settings.mis_heuristic));
    hash_value(hash, std::max(2, settings.irradiance_volume_grid_resolution));
    hash_value(hash, std::max(2, settings.irradiance_volume_subgrid_resolution));
    hash_value(hash, std::max(1, settings.irradiance_volume_direction_resolution));
    hash_value(hash, std::max(1, settings.irradiance_volume_bake_samples));
    hash_value(hash, std::max(1, settings.irradiance_volume_bake_bounces));
    hash_value(hash, std::clamp(settings.irradiance_volume_bounds_inset, 0.0f, 0.45f));
    hash_value(hash, settings.irradiance_volume_manual_bounds);
    hash_vec3(hash, settings.irradiance_volume_bounds_min);
    hash_vec3(hash, settings.irradiance_volume_bounds_max);

    const uint64_t triangle_count = static_cast<uint64_t>(render_scene.triangles.size());
    const uint64_t sphere_count = static_cast<uint64_t>(render_scene.spheres.size());
    hash_value(hash, triangle_count);
    for (const Triangle& triangle : render_scene.triangles) {
        hash_vec3(hash, triangle.v0);
        hash_vec3(hash, triangle.v1);
        hash_vec3(hash, triangle.v2);
        hash_vec3(hash, triangle.normal);
        hash_vec3(hash, triangle.n0);
        hash_vec3(hash, triangle.n1);
        hash_vec3(hash, triangle.n2);
        hash_vec3(hash, triangle.tangent);
        hash_vec3(hash, triangle.bitangent);
        hash_vec2(hash, triangle.uv0);
        hash_vec2(hash, triangle.uv1);
        hash_vec2(hash, triangle.uv2);
        hash_value(hash, triangle.material);
        hash_value(hash, triangle.mesh);
    }
    hash_value(hash, sphere_count);
    for (const RenderSphere& sphere : render_scene.spheres) {
        hash_vec3(hash, sphere.center);
        hash_value(hash, sphere.radius);
        hash_value(hash, sphere.material);
        hash_value(hash, sphere.sphere);
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

bool scene_has_irradiance_volume_exclusions(const Scene& scene) {
    for (const Mesh& mesh : scene.meshes) {
        if (mesh.exclude_from_irradiance_volume_bake) return true;
    }
    for (const Sphere& sphere : scene.spheres) {
        if (sphere.exclude_from_irradiance_volume_bake) return true;
    }
    return false;
}

Scene make_irradiance_volume_bake_scene(const Scene& scene) {
    Scene bake_scene = scene;
    bake_scene.meshes.erase(
        std::remove_if(
            bake_scene.meshes.begin(),
            bake_scene.meshes.end(),
            [](const Mesh& mesh) { return mesh.exclude_from_irradiance_volume_bake; }),
        bake_scene.meshes.end());
    bake_scene.spheres.erase(
        std::remove_if(
            bake_scene.spheres.begin(),
            bake_scene.spheres.end(),
            [](const Sphere& sphere) { return sphere.exclude_from_irradiance_volume_bake; }),
        bake_scene.spheres.end());
    return bake_scene;
}

std::string irradiance_volume_cache_path(const RenderSettings& settings) {
    if (settings.irradiance_volume_cache_path[0] != '\0') {
        return settings.irradiance_volume_cache_path;
    }
    if (settings.irradiance_volume_cache_key[0] != '\0') {
        return std::string(settings.irradiance_volume_cache_key) + ".ivol";
    }
    return {};
}

template <typename T>
bool write_binary(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(output);
}

template <typename T>
bool read_binary(std::istream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

uint64_t expected_grid_samples(int resolution) {
    const uint64_t r = static_cast<uint64_t>(std::max(0, resolution));
    return r * r * r;
}

uint64_t expected_grid_cells(int resolution) {
    const uint64_t r = static_cast<uint64_t>(std::max(0, resolution - 1));
    return r * r * r;
}

bool write_irradiance_cache_grid(std::ostream& output, const IrradianceVolumeGrid& grid, size_t direction_count) {
    const uint64_t sample_count = static_cast<uint64_t>(grid.samples.size());
    const uint64_t cell_count = static_cast<uint64_t>(grid.cells.size());
    if (!write_binary(output, grid.bounds.min) ||
        !write_binary(output, grid.bounds.max) ||
        !write_binary(output, grid.resolution) ||
        !write_binary(output, sample_count) ||
        !write_binary(output, cell_count)) {
        return false;
    }
    for (const IrradianceVolumeSample& sample : grid.samples) {
        if (!write_binary(output, sample.position)) {
            return false;
        }
        for (size_t i = 0; i < direction_count; ++i) {
            const Vec3 value = i < sample.irradiance.size() ? sample.irradiance[i] : Vec3{};
            if (!write_binary(output, value)) {
                return false;
            }
        }
    }
    for (const IrradianceVolumeCell& cell : grid.cells) {
        const uint8_t has_subgrid = cell.subgrid ? 1u : 0u;
        if (!write_binary(output, has_subgrid)) {
            return false;
        }
        if (cell.subgrid && !write_irradiance_cache_grid(output, *cell.subgrid, direction_count)) {
            return false;
        }
    }
    return true;
}

bool read_irradiance_cache_grid(
    std::istream& input,
    IrradianceVolume& volume,
    IrradianceVolumeGrid& grid,
    size_t direction_count,
    bool root_grid) {
    grid = {};
    uint64_t sample_count = 0;
    uint64_t cell_count = 0;
    if (!read_binary(input, grid.bounds.min) ||
        !read_binary(input, grid.bounds.max) ||
        !read_binary(input, grid.resolution) ||
        !read_binary(input, sample_count) ||
        !read_binary(input, cell_count)) {
        return false;
    }
    if (grid.resolution < 2 || grid.resolution > 256 ||
        sample_count != expected_grid_samples(grid.resolution) ||
        (cell_count != 0u && cell_count != expected_grid_cells(grid.resolution)) ||
        direction_count > 16384u ||
        !aabb_is_valid(grid.bounds)) {
        return false;
    }
    grid.samples.resize(static_cast<size_t>(sample_count));
    for (IrradianceVolumeSample& sample : grid.samples) {
        if (!read_binary(input, sample.position)) {
            return false;
        }
        sample.irradiance.resize(direction_count);
        for (Vec3& value : sample.irradiance) {
            if (!read_binary(input, value)) {
                return false;
            }
        }
    }
    grid.cells.resize(static_cast<size_t>(cell_count));
    volume.spatial_sample_count += static_cast<size_t>(sample_count);
    if (root_grid) {
        volume.first_level_cell_count = static_cast<size_t>(cell_count);
    }
    for (IrradianceVolumeCell& cell : grid.cells) {
        uint8_t has_subgrid = 0;
        if (!read_binary(input, has_subgrid)) {
            return false;
        }
        if (has_subgrid) {
            cell.subgrid = std::make_unique<IrradianceVolumeGrid>();
            ++volume.subgrid_count;
            if (!read_irradiance_cache_grid(input, volume, *cell.subgrid, direction_count, false)) {
                return false;
            }
        }
    }
    return true;
}

std::shared_ptr<IrradianceVolume> load_irradiance_volume_cache(
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings,
    uint64_t fingerprint) {
    const std::string path = irradiance_volume_cache_path(settings);
    if (!settings.irradiance_volume_cache_enabled || path.empty()) {
        return nullptr;
    }
    reset_irradiance_volume_progress(settings, IrradianceVolumeBakePhase::LoadingCache);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Idle);
        return nullptr;
    }

    const auto begin = std::chrono::steady_clock::now();
    IrradianceVolumeCacheHeader header;
    if (!read_binary(input, header) ||
        std::memcmp(header.magic, kIrradianceVolumeCacheMagic, sizeof(header.magic)) != 0 ||
        header.version != kIrradianceVolumeCacheVersion ||
        header.header_size != sizeof(IrradianceVolumeCacheHeader) ||
        header.fingerprint != fingerprint) {
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Idle);
        return nullptr;
    }

    const Aabb expected_bounds = scene_irradiance_bounds(render_scene, settings);
    const Aabb cached_bounds{header.bounds_min, header.bounds_max};
    if (!aabb_is_valid(cached_bounds) || !nearly_equal_bounds(cached_bounds, expected_bounds)) {
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Idle);
        LT_LOG_WARN("Irradiance volume cache ignored: path='{}' reason='bounds mismatch'", path);
        return nullptr;
    }

    auto volume = std::make_shared<IrradianceVolume>();
    volume->grid_resolution = static_cast<int>(header.grid_resolution);
    volume->subgrid_resolution = static_cast<int>(header.subgrid_resolution);
    volume->direction_resolution = static_cast<int>(header.direction_resolution);
    volume->bake_samples = static_cast<int>(header.bake_samples);
    volume->bake_bounces = static_cast<int>(header.bake_bounces);
    volume->bounds = {header.bounds_min, header.bounds_max};
    volume->directions = make_irradiance_volume_directions(volume->direction_resolution);
    volume->cosine_weights = make_irradiance_volume_weights(volume->directions);
    if (volume->grid_resolution < 2 ||
        volume->subgrid_resolution < 2 ||
        volume->direction_resolution < 1 ||
        volume->directions.empty() ||
        !aabb_is_valid(volume->bounds) ||
        !read_irradiance_cache_grid(input, *volume, volume->grid, volume->directions.size(), true)) {
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Failed);
        LT_LOG_WARN("Irradiance volume cache ignored: path='{}' reason='invalid cache payload'", path);
        return nullptr;
    }
    collect_debug_probes_from_grid(*volume, volume->grid);
    volume->unique_debug_probe_count = volume->debug_probes.size();

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    if (settings.irradiance_volume_bake_progress) {
        settings.irradiance_volume_bake_progress->total_samples.store(volume->spatial_sample_count, std::memory_order_relaxed);
        settings.irradiance_volume_bake_progress->completed_samples.store(volume->spatial_sample_count, std::memory_order_relaxed);
        settings.irradiance_volume_bake_progress->direction_count.store(static_cast<int>(volume->directions.size()), std::memory_order_relaxed);
        settings.irradiance_volume_bake_progress->elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
        settings.irradiance_volume_bake_progress->phase.store(static_cast<int>(IrradianceVolumeBakePhase::Complete), std::memory_order_relaxed);
    }
    LT_LOG_INFO(
        "Irradiance volume cache loaded: path='{}' samples={} debug_probes={} first_cells={} subgrids={} directions={} elapsed_ms={} elapsed_s={}",
        path,
        volume->spatial_sample_count,
        volume->unique_debug_probe_count,
        volume->first_level_cell_count,
        volume->subgrid_count,
        volume->directions.size(),
        format_decimal(elapsed_ms, 3),
        format_decimal(elapsed_ms * 0.001, 2));
    (void)scene;
    return volume;
}

bool save_irradiance_volume_cache(
    const IrradianceVolume& volume,
    const RenderSettings& settings,
    uint64_t fingerprint) {
    const std::string path = irradiance_volume_cache_path(settings);
    if (!settings.irradiance_volume_cache_enabled || path.empty()) {
        return false;
    }
    set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::SavingCache);
    std::error_code error;
    const std::filesystem::path cache_path(path);
    const std::filesystem::path parent = cache_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            LT_LOG_WARN("Could not create irradiance volume cache directory '{}': {}", parent.string(), error.message());
            set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Complete);
            return false;
        }
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        LT_LOG_WARN("Could not write irradiance volume cache '{}'", path);
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Complete);
        return false;
    }
    IrradianceVolumeCacheHeader header;
    std::memcpy(header.magic, kIrradianceVolumeCacheMagic, sizeof(header.magic));
    header.version = kIrradianceVolumeCacheVersion;
    header.header_size = sizeof(IrradianceVolumeCacheHeader);
    header.fingerprint = fingerprint;
    header.grid_resolution = static_cast<uint32_t>(volume.grid_resolution);
    header.subgrid_resolution = static_cast<uint32_t>(volume.subgrid_resolution);
    header.direction_resolution = static_cast<uint32_t>(volume.direction_resolution);
    header.bake_samples = static_cast<uint32_t>(volume.bake_samples);
    header.bake_bounces = static_cast<uint32_t>(volume.bake_bounces);
    header.bounds_min = volume.bounds.min;
    header.bounds_max = volume.bounds.max;
    if (!write_binary(output, header) ||
        !write_irradiance_cache_grid(output, volume.grid, volume.directions.size())) {
        LT_LOG_WARN("Could not write complete irradiance volume cache '{}'", path);
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Complete);
        return false;
    }
    LT_LOG_INFO("Irradiance volume cache saved: path='{}'", path);
    set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Complete);
    return true;
}

void initialize_irradiance_grid(
    IrradianceVolume& volume,
    IrradianceVolumeGrid& grid,
    const RenderScene& render_scene,
    const Aabb& bounds,
    int resolution,
    bool build_subgrids) {
    resolution = std::max(2, resolution);
    grid.bounds = bounds;
    grid.resolution = resolution;
    grid.samples.clear();
    grid.cells.clear();
    const size_t sample_count = static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * static_cast<size_t>(resolution);
    grid.samples.resize(sample_count);
    for (int z = 0; z < resolution; ++z) {
        for (int y = 0; y < resolution; ++y) {
            for (int x = 0; x < resolution; ++x) {
                IrradianceVolumeSample& sample = grid.samples[static_cast<size_t>(grid_sample_index(resolution, x, y, z))];
                sample.position = grid_position(bounds, resolution, x, y, z);
                sample.irradiance.assign(volume.directions.size(), Vec3{});
            }
        }
    }
    volume.spatial_sample_count += sample_count;

    if (!build_subgrids) {
        return;
    }

    const int cells_per_axis = resolution - 1;
    const size_t cell_count = static_cast<size_t>(cells_per_axis) * static_cast<size_t>(cells_per_axis) * static_cast<size_t>(cells_per_axis);
    grid.cells.resize(cell_count);
    volume.first_level_cell_count = cell_count;
    for (int z = 0; z < cells_per_axis; ++z) {
        for (int y = 0; y < cells_per_axis; ++y) {
            for (int x = 0; x < cells_per_axis; ++x) {
                const Aabb cell_bounds = grid_cell_bounds(bounds, resolution, x, y, z);
                if (!cell_contains_irradiance_geometry(render_scene, cell_bounds)) {
                    continue;
                }
                IrradianceVolumeCell& cell = grid.cells[static_cast<size_t>(grid_cell_index(resolution, x, y, z))];
                cell.subgrid = std::make_unique<IrradianceVolumeGrid>();
                ++volume.subgrid_count;
                initialize_irradiance_grid(
                    volume,
                    *cell.subgrid,
                    render_scene,
                    cell_bounds,
                    volume.subgrid_resolution,
                    false);
            }
        }
    }
}

Vec3 trace_volume_radiance(
    const RenderScene& render_scene,
    const Scene& scene,
    Vec3 origin,
    Vec3 direction,
    Rng& rng,
    RenderSettings bake_settings) {
    bake_settings.use_irradiance_volume = false;
    bake_settings.stylized_samples = 0;
    bake_settings.stylized_max_depth = 0;
    bake_settings.max_bounces = std::max(1, bake_settings.irradiance_volume_bake_bounces);

    Ray probe_ray{origin + direction * 0.002f, direction};
    Rng visibility_rng = rng;
    for (int step = 0; step < 8; ++step) {
        Hit first_hit;
        if (!intersect_scene(render_scene, probe_ray, first_hit, bake_settings.acceleration_structure)) {
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
            probe_ray = {first_hit.position + direction * 0.002f, direction};
            continue;
        }
        Vec3 emission;
        if (first_hit.triangle >= 0 && first_hit.triangle < static_cast<int>(render_scene.triangles.size())) {
            emission = emitted_radiance(scene, render_scene.triangles[static_cast<size_t>(first_hit.triangle)], first_hit.uv, direction);
        }
        if (has_light_emission(emission)) {
            return {};
        }
        break;
    }
    return trace_path(render_scene, scene, probe_ray, rng, bake_settings, nullptr);
}

void bake_irradiance_sample(
    IrradianceVolume& volume,
    IrradianceVolumeSample& sample,
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings,
    uint32_t sample_serial) {
    const size_t direction_count = volume.directions.size();
    sample.irradiance.assign(direction_count, Vec3{});
    if (direction_count == 0) {
        return;
    }

    std::vector<Vec3> radiance(direction_count, Vec3{});
    const int bake_samples = std::max(1, volume.bake_samples);
    uint64_t traced_rays = 0;
    for (size_t direction_index = 0; direction_index < direction_count; ++direction_index) {
        const Vec3 direction = volume.directions[direction_index];
        Vec3 estimate;
        for (int sample_index = 0; sample_index < bake_samples; ++sample_index) {
            const uint32_t seed = hash_u32(
                sample_serial * 0x8da6b343u ^
                static_cast<uint32_t>(direction_index) * 0xd8163841u ^
                static_cast<uint32_t>(sample_index) * 0xcb1ab31fu ^
                0x9e3779b9u);
            Rng rng(seed);
            estimate += trace_volume_radiance(render_scene, scene, sample.position, direction, rng, settings);
            ++volume.radiance_trace_count;
            ++traced_rays;
        }
        radiance[direction_index] = estimate / static_cast<float>(bake_samples);
    }

    for (size_t normal_index = 0; normal_index < direction_count; ++normal_index) {
        Vec3 irradiance;
        const size_t weight_offset = normal_index * direction_count;
        for (size_t radiance_index = 0; radiance_index < direction_count; ++radiance_index) {
            irradiance += radiance[radiance_index] * volume.cosine_weights[weight_offset + radiance_index];
        }
        sample.irradiance[normal_index] = clamp_sample_radiance(irradiance, 1024.0f);
    }

    if (settings.irradiance_volume_bake_progress) {
        settings.irradiance_volume_bake_progress->traced_rays.fetch_add(traced_rays, std::memory_order_relaxed);
        settings.irradiance_volume_bake_progress->completed_samples.fetch_add(1, std::memory_order_relaxed);
    }
}

void bake_irradiance_grid(
    IrradianceVolume& volume,
    IrradianceVolumeGrid& grid,
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings,
    uint32_t& sample_serial) {
    for (IrradianceVolumeSample& sample : grid.samples) {
        bake_irradiance_sample(volume, sample, render_scene, scene, settings, sample_serial++);
    }
    for (IrradianceVolumeCell& cell : grid.cells) {
        if (cell.subgrid) {
            bake_irradiance_grid(volume, *cell.subgrid, render_scene, scene, settings, sample_serial);
        }
    }
}

std::shared_ptr<IrradianceVolume> build_irradiance_volume(
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings) {
    const auto begin = std::chrono::steady_clock::now();
    auto volume = std::make_shared<IrradianceVolume>();
    volume->grid_resolution = std::max(2, settings.irradiance_volume_grid_resolution);
    volume->subgrid_resolution = std::max(2, settings.irradiance_volume_subgrid_resolution);
    volume->direction_resolution = std::max(1, settings.irradiance_volume_direction_resolution);
    volume->bake_samples = std::max(1, settings.irradiance_volume_bake_samples);
    volume->bake_bounces = std::max(1, settings.irradiance_volume_bake_bounces);
    volume->bounds = scene_irradiance_bounds(render_scene, settings);
    volume->directions = make_irradiance_volume_directions(volume->direction_resolution);
    volume->cosine_weights = make_irradiance_volume_weights(volume->directions);

    initialize_irradiance_grid(*volume, volume->grid, render_scene, volume->bounds, volume->grid_resolution, true);
    collect_debug_probes_from_grid(*volume, volume->grid);
    volume->unique_debug_probe_count = volume->debug_probes.size();

    if (settings.irradiance_volume_bake_progress) {
        IrradianceVolumeBakeProgress& progress = *settings.irradiance_volume_bake_progress;
        progress.total_samples.store(volume->spatial_sample_count, std::memory_order_relaxed);
        progress.completed_samples.store(0, std::memory_order_relaxed);
        progress.total_rays.store(
            volume->spatial_sample_count * volume->directions.size() * static_cast<size_t>(volume->bake_samples),
            std::memory_order_relaxed);
        progress.traced_rays.store(0, std::memory_order_relaxed);
        progress.direction_count.store(static_cast<int>(volume->directions.size()), std::memory_order_relaxed);
        progress.elapsed_ms.store(0.0, std::memory_order_relaxed);
        progress.phase.store(static_cast<int>(IrradianceVolumeBakePhase::Baking), std::memory_order_relaxed);
    }

    uint32_t sample_serial = 1u;
    bake_irradiance_grid(*volume, volume->grid, render_scene, scene, settings, sample_serial);

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    if (settings.irradiance_volume_bake_progress) {
        settings.irradiance_volume_bake_progress->elapsed_ms.store(elapsed_ms, std::memory_order_relaxed);
        settings.irradiance_volume_bake_progress->phase.store(static_cast<int>(IrradianceVolumeBakePhase::Complete), std::memory_order_relaxed);
    }
    const size_t irradiance_bytes = volume->spatial_sample_count * volume->directions.size() * sizeof(Vec3);
    const size_t weight_bytes = volume->cosine_weights.size() * sizeof(float);
    LT_LOG_INFO(
        "Irradiance volume baked: samples={} debug_probes={} first_cells={} subgrids={} directions={} rays={} memory_kib={} elapsed_ms={} elapsed_s={}",
        volume->spatial_sample_count,
        volume->unique_debug_probe_count,
        volume->first_level_cell_count,
        volume->subgrid_count,
        volume->directions.size(),
        volume->radiance_trace_count,
        (irradiance_bytes + weight_bytes) / 1024u,
        format_decimal(elapsed_ms, 3),
        format_decimal(elapsed_ms * 0.001, 2));
    return volume;
}

std::shared_ptr<IrradianceVolume> update_irradiance_volume(
    std::shared_ptr<void>& cached_irradiance_volume,
    const RenderScene& render_scene,
    const Scene& scene,
    const RenderSettings& settings,
    bool volume_dirty,
    bool& volume_rebuilt) {
    volume_rebuilt = false;
    const bool force_rebake = settings.irradiance_volume_force_rebake;
    const bool needs_update = !cached_irradiance_volume || volume_dirty || force_rebake;
    if (!needs_update) {
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Idle);
        return std::static_pointer_cast<IrradianceVolume>(cached_irradiance_volume);
    }

    Scene bake_scene_storage;
    RenderScene bake_render_scene_storage;
    const Scene* bake_scene = &scene;
    const RenderScene* bake_render_scene = &render_scene;
    if (scene_has_irradiance_volume_exclusions(scene)) {
        bake_scene_storage = make_irradiance_volume_bake_scene(scene);
        bake_render_scene_storage = build_render_scene(bake_scene_storage);
        bake_scene = &bake_scene_storage;
        bake_render_scene = &bake_render_scene_storage;
    }

    const uint64_t fingerprint = irradiance_volume_fingerprint(*bake_render_scene, *bake_scene, settings);
    if (!force_rebake) {
        if (std::shared_ptr<IrradianceVolume> cached = load_irradiance_volume_cache(*bake_render_scene, *bake_scene, settings, fingerprint)) {
            cached_irradiance_volume = cached;
            volume_rebuilt = true;
            return cached;
        }
    }

    if (cached_irradiance_volume && !settings.irradiance_volume_auto_update && !force_rebake) {
        LT_LOG_INFO("Irradiance volume auto update disabled; keeping existing in-memory volume");
        set_irradiance_volume_progress_phase(settings, IrradianceVolumeBakePhase::Idle);
        return std::static_pointer_cast<IrradianceVolume>(cached_irradiance_volume);
    }
    if (!settings.irradiance_volume_auto_update && !force_rebake) {
        LT_LOG_WARN("Irradiance volume auto update disabled, but no usable cached volume exists; baking initial volume");
    }

    reset_irradiance_volume_progress(settings, IrradianceVolumeBakePhase::Baking);
    std::shared_ptr<IrradianceVolume> volume = build_irradiance_volume(*bake_render_scene, *bake_scene, settings);
    cached_irradiance_volume = volume;
    volume_rebuilt = true;
    save_irradiance_volume_cache(*volume, settings, fingerprint);
    return volume;
}

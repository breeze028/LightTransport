struct IrradianceVolumeGrid;

struct IrradianceVolumeSample {
    Vec3 position;
    std::vector<Vec3> irradiance;
};

struct IrradianceVolumeCell {
    std::unique_ptr<IrradianceVolumeGrid> subgrid;
};

struct IrradianceVolumeDebugProbe {
    Vec3 position;
    float radius = 0.05f;
};

struct IrradianceVolumeGrid {
    Aabb bounds;
    int resolution = 0;
    std::vector<IrradianceVolumeSample> samples;
    std::vector<IrradianceVolumeCell> cells;
};

struct IrradianceVolume {
    Aabb bounds;
    int grid_resolution = 0;
    int subgrid_resolution = 0;
    int direction_resolution = 0;
    int bake_samples = 0;
    int bake_bounces = 0;
    std::vector<Vec3> directions;
    std::vector<float> cosine_weights;
    std::vector<IrradianceVolumeDebugProbe> debug_probes;
    IrradianceVolumeGrid grid;
    size_t spatial_sample_count = 0;
    size_t unique_debug_probe_count = 0;
    size_t first_level_cell_count = 0;
    size_t subgrid_count = 0;
    size_t radiance_trace_count = 0;
};

int grid_sample_index(int resolution, int x, int y, int z) {
    return (z * resolution + y) * resolution + x;
}

int grid_cell_index(int resolution, int x, int y, int z) {
    const int cells_per_axis = resolution - 1;
    return (z * cells_per_axis + y) * cells_per_axis + x;
}

float axis_component(Vec3 value, int axis) {
    return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
}

Vec3 with_axis(Vec3 value, int axis, float component) {
    if (axis == 0) {
        value.x = component;
    } else if (axis == 1) {
        value.y = component;
    } else {
        value.z = component;
    }
    return value;
}

Vec3 lerp_vec3(Vec3 a, Vec3 b, float t) {
    return a * (1.0f - t) + b * t;
}

Vec3 grid_position(const Aabb& bounds, int resolution, int x, int y, int z) {
    const float inv = resolution > 1 ? 1.0f / static_cast<float>(resolution - 1) : 0.0f;
    const Vec3 extent = bounds.max - bounds.min;
    return {
        bounds.min.x + extent.x * static_cast<float>(x) * inv,
        bounds.min.y + extent.y * static_cast<float>(y) * inv,
        bounds.min.z + extent.z * static_cast<float>(z) * inv,
    };
}

Aabb grid_cell_bounds(const Aabb& bounds, int resolution, int x, int y, int z) {
    Aabb cell;
    cell.min = grid_position(bounds, resolution, x, y, z);
    cell.max = grid_position(bounds, resolution, x + 1, y + 1, z + 1);
    return cell;
}

bool aabb_is_valid(const Aabb& bounds) {
    return std::isfinite(bounds.min.x) && std::isfinite(bounds.min.y) && std::isfinite(bounds.min.z) &&
        std::isfinite(bounds.max.x) && std::isfinite(bounds.max.y) && std::isfinite(bounds.max.z) &&
        bounds.max.x > bounds.min.x && bounds.max.y > bounds.min.y && bounds.max.z > bounds.min.z;
}

bool aabb_overlaps(const Aabb& a, const Aabb& b) {
    return a.min.x <= b.max.x && a.max.x >= b.min.x &&
        a.min.y <= b.max.y && a.max.y >= b.min.y &&
        a.min.z <= b.max.z && a.max.z >= b.min.z;
}

Vec2 concentric_sample_disk(float u, float v) {
    const float sx = 2.0f * u - 1.0f;
    const float sy = 2.0f * v - 1.0f;
    if (sx == 0.0f && sy == 0.0f) {
        return {};
    }

    float r = 0.0f;
    float theta = 0.0f;
    if (std::fabs(sx) > std::fabs(sy)) {
        r = sx;
        theta = (kPi * 0.25f) * (sy / sx);
    } else {
        r = sy;
        theta = (kPi * 0.5f) - (kPi * 0.25f) * (sx / sy);
    }
    return {r * std::cos(theta), r * std::sin(theta)};
}

Vec3 equal_area_hemisphere_direction(float u, float v, bool upper) {
    const Vec2 disk = concentric_sample_disk(u, v);
    const float rho2 = std::clamp(disk.x * disk.x + disk.y * disk.y, 0.0f, 1.0f);
    const float z_abs = 1.0f - rho2;
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - z_abs * z_abs));
    const float rho = std::sqrt(rho2);
    Vec3 direction;
    if (rho > 1.0e-8f) {
        direction = {disk.x / rho * sin_theta, disk.y / rho * sin_theta, upper ? z_abs : -z_abs};
    } else {
        direction = {0.0f, 0.0f, upper ? 1.0f : -1.0f};
    }
    return normalize(direction);
}

std::vector<Vec3> make_irradiance_volume_directions(int resolution) {
    resolution = std::max(1, resolution);
    std::vector<Vec3> directions;
    directions.reserve(static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * 2u);
    for (int hemisphere = 0; hemisphere < 2; ++hemisphere) {
        const bool upper = hemisphere == 0;
        for (int y = 0; y < resolution; ++y) {
            for (int x = 0; x < resolution; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(resolution);
                const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(resolution);
                directions.push_back(equal_area_hemisphere_direction(u, v, upper));
            }
        }
    }
    return directions;
}

std::vector<float> make_irradiance_volume_weights(const std::vector<Vec3>& directions) {
    const size_t direction_count = directions.size();
    std::vector<float> weights(direction_count * direction_count, 0.0f);
    if (direction_count == 0) {
        return weights;
    }
    const float scale = 4.0f * kPi / static_cast<float>(direction_count);
    for (size_t normal_index = 0; normal_index < direction_count; ++normal_index) {
        for (size_t radiance_index = 0; radiance_index < direction_count; ++radiance_index) {
            weights[normal_index * direction_count + radiance_index] =
                scale * std::max(0.0f, dot(directions[radiance_index], directions[normal_index]));
        }
    }
    return weights;
}

int nearest_irradiance_direction(const IrradianceVolume& volume, Vec3 direction) {
    direction = normalize(direction);
    if (dot(direction, direction) <= 0.0f || volume.directions.empty()) {
        return -1;
    }
    int best = 0;
    float best_dot = -kInfinity;
    for (int i = 0; i < static_cast<int>(volume.directions.size()); ++i) {
        const float d = dot(direction, volume.directions[static_cast<size_t>(i)]);
        if (d > best_dot) {
            best_dot = d;
            best = i;
        }
    }
    return best;
}

Vec3 sample_irradiance_value(const IrradianceVolumeSample& sample, int direction_index) {
    if (direction_index < 0 || direction_index >= static_cast<int>(sample.irradiance.size())) {
        return {};
    }
    return sample.irradiance[static_cast<size_t>(direction_index)];
}

Vec3 query_irradiance_grid(const IrradianceVolumeGrid& grid, int direction_index, Vec3 position) {
    if (grid.resolution < 2 || grid.samples.empty() || !aabb_is_valid(grid.bounds)) {
        return {};
    }

    const Vec3 extent = grid.bounds.max - grid.bounds.min;
    const float sx = std::clamp((position.x - grid.bounds.min.x) / std::max(1.0e-8f, extent.x), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const float sy = std::clamp((position.y - grid.bounds.min.y) / std::max(1.0e-8f, extent.y), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const float sz = std::clamp((position.z - grid.bounds.min.z) / std::max(1.0e-8f, extent.z), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, grid.resolution - 2);
    const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, grid.resolution - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(sz)), 0, grid.resolution - 2);
    const float fx = std::clamp(sx - static_cast<float>(x0), 0.0f, 1.0f);
    const float fy = std::clamp(sy - static_cast<float>(y0), 0.0f, 1.0f);
    const float fz = std::clamp(sz - static_cast<float>(z0), 0.0f, 1.0f);

    const auto sample_at = [&](int x, int y, int z) {
        const int index = grid_sample_index(grid.resolution, x, y, z);
        if (index < 0 || index >= static_cast<int>(grid.samples.size())) {
            return Vec3{};
        }
        return sample_irradiance_value(grid.samples[static_cast<size_t>(index)], direction_index);
    };

    const Vec3 c000 = sample_at(x0, y0, z0);
    const Vec3 c100 = sample_at(x0 + 1, y0, z0);
    const Vec3 c010 = sample_at(x0, y0 + 1, z0);
    const Vec3 c110 = sample_at(x0 + 1, y0 + 1, z0);
    const Vec3 c001 = sample_at(x0, y0, z0 + 1);
    const Vec3 c101 = sample_at(x0 + 1, y0, z0 + 1);
    const Vec3 c011 = sample_at(x0, y0 + 1, z0 + 1);
    const Vec3 c111 = sample_at(x0 + 1, y0 + 1, z0 + 1);
    const Vec3 c00 = lerp_vec3(c000, c100, fx);
    const Vec3 c10 = lerp_vec3(c010, c110, fx);
    const Vec3 c01 = lerp_vec3(c001, c101, fx);
    const Vec3 c11 = lerp_vec3(c011, c111, fx);
    const Vec3 c0 = lerp_vec3(c00, c10, fy);
    const Vec3 c1 = lerp_vec3(c01, c11, fy);
    return lerp_vec3(c0, c1, fz);
}

const IrradianceVolumeGrid* query_grid_for_position(const IrradianceVolumeGrid& grid, Vec3 position) {
    if (grid.resolution < 2 || !aabb_is_valid(grid.bounds) || grid.cells.empty()) {
        return &grid;
    }
    const Vec3 extent = grid.bounds.max - grid.bounds.min;
    const float sx = std::clamp((position.x - grid.bounds.min.x) / std::max(1.0e-8f, extent.x), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const float sy = std::clamp((position.y - grid.bounds.min.y) / std::max(1.0e-8f, extent.y), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const float sz = std::clamp((position.z - grid.bounds.min.z) / std::max(1.0e-8f, extent.z), 0.0f, 1.0f) * static_cast<float>(grid.resolution - 1);
    const int x = std::clamp(static_cast<int>(std::floor(sx)), 0, grid.resolution - 2);
    const int y = std::clamp(static_cast<int>(std::floor(sy)), 0, grid.resolution - 2);
    const int z = std::clamp(static_cast<int>(std::floor(sz)), 0, grid.resolution - 2);
    const int cell = grid_cell_index(grid.resolution, x, y, z);
    if (cell >= 0 && cell < static_cast<int>(grid.cells.size()) && grid.cells[static_cast<size_t>(cell)].subgrid) {
        return grid.cells[static_cast<size_t>(cell)].subgrid.get();
    }
    return &grid;
}

Vec3 query_irradiance_volume(const IrradianceVolume& volume, Vec3 position, Vec3 normal) {
    const int direction_index = nearest_irradiance_direction(volume, normal);
    if (direction_index < 0) {
        return {};
    }
    const IrradianceVolumeGrid* grid = query_grid_for_position(volume.grid, position);
    if (!grid) {
        return {};
    }
    return query_irradiance_grid(*grid, direction_index, position);
}

float debug_probe_grid_radius(const IrradianceVolumeGrid& grid) {
    if (grid.resolution < 2 || !aabb_is_valid(grid.bounds)) {
        return 0.05f;
    }
    const Vec3 spacing = (grid.bounds.max - grid.bounds.min) / static_cast<float>(grid.resolution - 1);
    const float min_spacing = std::min({std::fabs(spacing.x), std::fabs(spacing.y), std::fabs(spacing.z)});
    return std::max(1.0e-4f, min_spacing);
}

bool debug_probe_already_exists(const std::vector<IrradianceVolumeDebugProbe>& probes, Vec3 position, float epsilon) {
    const float epsilon2 = epsilon * epsilon;
    for (const IrradianceVolumeDebugProbe& probe : probes) {
        const Vec3 d = probe.position - position;
        if (dot(d, d) <= epsilon2) {
            return true;
        }
    }
    return false;
}

void collect_debug_probes_from_grid(IrradianceVolume& volume, const IrradianceVolumeGrid& grid) {
    const float radius = debug_probe_grid_radius(grid);
    const float epsilon = std::max(1.0e-5f, radius * 1.0e-3f);
    for (const IrradianceVolumeSample& sample : grid.samples) {
        if (!debug_probe_already_exists(volume.debug_probes, sample.position, epsilon)) {
            volume.debug_probes.push_back({sample.position, radius});
        }
    }
    for (const IrradianceVolumeCell& cell : grid.cells) {
        if (cell.subgrid) {
            collect_debug_probes_from_grid(volume, *cell.subgrid);
        }
    }
}

struct IrradianceVolumeProbeHit {
    float t = kInfinity;
    Vec3 position;
    Vec3 normal;
};

bool intersect_debug_probe_sphere(const IrradianceVolumeDebugProbe& probe, const Ray& ray, float radius_scale, IrradianceVolumeProbeHit& hit) {
    const float radius = std::max(1.0e-5f, probe.radius * std::max(0.0f, radius_scale));
    const Vec3 oc = ray.origin - probe.position;
    const float a = dot(ray.direction, ray.direction);
    const float half_b = dot(oc, ray.direction);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float root = std::sqrt(discriminant);
    float t = (-half_b - root) / a;
    if (t <= 0.001f || t >= hit.t) {
        t = (-half_b + root) / a;
        if (t <= 0.001f || t >= hit.t) {
            return false;
        }
    }
    hit.t = t;
    hit.position = ray.origin + ray.direction * t;
    hit.normal = normalize((hit.position - probe.position) / radius);
    return true;
}

bool intersect_irradiance_debug_probes(const IrradianceVolume& volume, const Ray& ray, float radius_scale, IrradianceVolumeProbeHit& hit) {
    if (radius_scale <= 0.0f) {
        return false;
    }
    bool found = false;
    for (const IrradianceVolumeDebugProbe& probe : volume.debug_probes) {
        found = intersect_debug_probe_sphere(probe, ray, radius_scale, hit) || found;
    }
    return found;
}

Vec3 shade_irradiance_debug_probe(const IrradianceVolumeProbeHit& hit, Vec3 view_direction) {
    const Vec3 view_light = normalize(Vec3{-0.35f, 0.55f, -0.75f});
    const float lambert = std::max(0.0f, dot(hit.normal, -view_light));
    const float rim = std::pow(std::clamp(1.0f - std::fabs(dot(hit.normal, -view_direction)), 0.0f, 1.0f), 2.0f);
    const float tone = std::clamp(0.28f + lambert * 0.55f + rim * 0.18f, 0.0f, 1.0f);
    return Vec3{tone};
}

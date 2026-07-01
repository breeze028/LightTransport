struct LightmapTriangle {
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    bool valid = false;
};

struct Lightmap {
    int width = 0;
    int height = 0;
    int bake_samples = 0;
    int bake_bounces = 0;
    std::vector<Vec3> texels;
    std::vector<uint8_t> valid;
    std::vector<LightmapTriangle> triangles;
    size_t baked_texel_count = 0;
    size_t traced_ray_count = 0;
};

int lightmap_texel_index(const Lightmap& lightmap, int x, int y) {
    return y * lightmap.width + x;
}

Vec3 sample_lightmap_texel(const Lightmap& lightmap, int x, int y) {
    if (lightmap.width <= 0 || lightmap.height <= 0 || lightmap.texels.empty()) {
        return {};
    }
    x = std::clamp(x, 0, lightmap.width - 1);
    y = std::clamp(y, 0, lightmap.height - 1);
    const int index = lightmap_texel_index(lightmap, x, y);
    if (index < 0 || index >= static_cast<int>(lightmap.texels.size()) ||
        index >= static_cast<int>(lightmap.valid.size()) || !lightmap.valid[static_cast<size_t>(index)]) {
        return {};
    }
    return lightmap.texels[static_cast<size_t>(index)];
}

Vec3 query_lightmap(const Lightmap& lightmap, Vec2 uv) {
    if (lightmap.width <= 0 || lightmap.height <= 0 || lightmap.texels.empty()) {
        return {};
    }
    const float sx = std::clamp(uv.x, 0.0f, 1.0f) * static_cast<float>(lightmap.width - 1);
    const float sy = std::clamp(uv.y, 0.0f, 1.0f) * static_cast<float>(lightmap.height - 1);
    const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, lightmap.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, lightmap.height - 1);
    const int x1 = std::min(x0 + 1, lightmap.width - 1);
    const int y1 = std::min(y0 + 1, lightmap.height - 1);
    const float fx = std::clamp(sx - static_cast<float>(x0), 0.0f, 1.0f);
    const float fy = std::clamp(sy - static_cast<float>(y0), 0.0f, 1.0f);
    const Vec3 c00 = sample_lightmap_texel(lightmap, x0, y0);
    const Vec3 c10 = sample_lightmap_texel(lightmap, x1, y0);
    const Vec3 c01 = sample_lightmap_texel(lightmap, x0, y1);
    const Vec3 c11 = sample_lightmap_texel(lightmap, x1, y1);
    return lerp_vec3(lerp_vec3(c00, c10, fx), lerp_vec3(c01, c11, fx), fy);
}

void apply_lightmap_to_render_scene(const Lightmap* lightmap, RenderScene& render_scene) {
    for (Triangle& triangle : render_scene.triangles) {
        triangle.lightmap_uv0 = {};
        triangle.lightmap_uv1 = {};
        triangle.lightmap_uv2 = {};
        triangle.has_lightmap = false;
    }
    if (!lightmap) {
        return;
    }
    const size_t count = std::min(render_scene.triangles.size(), lightmap->triangles.size());
    for (size_t i = 0; i < count; ++i) {
        const LightmapTriangle& source = lightmap->triangles[i];
        if (!source.valid) {
            continue;
        }
        Triangle& target = render_scene.triangles[i];
        target.lightmap_uv0 = source.uv0;
        target.lightmap_uv1 = source.uv1;
        target.lightmap_uv2 = source.uv2;
        target.has_lightmap = true;
    }
}

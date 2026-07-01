
constexpr int kTraversalStackSize = 64;

struct Hit {
    float t = kInfinity;
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    Vec2 uv;
    Vec2 lightmap_uv;
    int material = -1;
    int mesh = -1;
    int triangle = -1;
    bool has_lightmap = false;
    bool front_face = true;
};

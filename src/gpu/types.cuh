struct GpuMaterial {
    Vec3 albedo;
    int brdf_model = 0;
    float roughness = 0.5f;
    float metallic = 0.0f;
    int texture_index = -1;
    int metallic_roughness_texture_index = -1;
    Vec3 sheen_color;
    float sheen_roughness = 0.0f;
    int sheen_color_texture_index = -1;
    int sheen_roughness_texture_index = -1;
    float clearcoat = 0.0f;
    float clearcoat_roughness = 0.0f;
    int clearcoat_texture_index = -1;
    int clearcoat_roughness_texture_index = -1;
    int normal_texture_index = -1;
    float normal_scale = 1.0f;
    Vec3 emission;
    int emission_texture_index = -1;
    Vec3 transmission_tint = {1.0f, 1.0f, 1.0f};
    float alpha = 1.0f;
    float alpha_cutoff = 0.5f;
    int alpha_mode = 0;
    int double_sided = 0;
};

struct GpuTexture {
    int width = 0;
    int height = 0;
    int mip_levels = 1;
    cudaTextureObject_t object = 0;
};

struct GpuDirectionalLight {
    Vec3 direction;
    Vec3 color;
    float intensity = 0.0f;
};

struct GpuTriangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
    Vec3 tangent;
    Vec3 bitangent;
    Vec3 centroid;
    Vec3 emission;
    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;
    int material = 0;
    int mesh = -1;
    int light_double_sided = 0;
};

struct GpuSphere {
    Vec3 center;
    float radius = 0.5f;
    int material = 0;
    int sphere = -1;
};

struct GpuBvhNode {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;
};

struct GpuMeshInstance {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int bvh_root = -1;
    int mesh = -1;
};

struct GpuIrradianceVolumeGrid {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int resolution = 0;
    int sample_offset = 0;
    int cell_offset = 0;
    int cell_count = 0;
};

struct GpuIrradianceVolumeDebugProbe {
    Vec3 position;
    float radius = 0.05f;
};

struct GpuIrradianceVolume {
    int direction_count = 0;
    int grid_count = 0;
    int irradiance_sample_count = 0;
    int cell_count = 0;
    int debug_probe_count = 0;
    Vec3* directions = nullptr;
    Vec3* irradiance = nullptr;
    GpuIrradianceVolumeGrid* grids = nullptr;
    int* cell_subgrid_indices = nullptr;
    GpuIrradianceVolumeDebugProbe* debug_probes = nullptr;
};

struct GpuScene {
    Camera camera;
    int material_count = 0;
    int triangle_count = 0;
    int sphere_count = 0;
    int bvh_node_count = 0;
    int tlas_node_count = 0;
    int mesh_instance_count = 0;
    int use_two_level = 0;
    int light_count = 0;
    int directional_light_count = 0;
    int texture_count = 0;
    int environment_texture = -1;
    Vec3 environment_color = {1.0f, 1.0f, 1.0f};
    float environment_strength = 1.0f;
    bool environment_constant = false;
    int environment_mapping = 0;
    Vec3 environment_light_from_world_x = {1.0f, 0.0f, 0.0f};
    Vec3 environment_light_from_world_y = {0.0f, 1.0f, 0.0f};
    Vec3 environment_light_from_world_z = {0.0f, 0.0f, 1.0f};
    GpuMaterial* materials = nullptr;
    GpuTexture* textures = nullptr;
    GpuTriangle* triangles = nullptr;
    GpuSphere* spheres = nullptr;
    int* triangle_indices = nullptr;
    int* light_indices = nullptr;
    GpuDirectionalLight* directional_lights = nullptr;
    GpuBvhNode* bvh_nodes = nullptr;
    GpuMeshInstance* mesh_instances = nullptr;
    int* mesh_instance_indices = nullptr;
    GpuBvhNode* tlas_nodes = nullptr;
    GpuIrradianceVolume irradiance_volume;
};

struct PackedGpuScene {
    GpuScene scene;
    std::vector<GpuMaterial> materials;
    std::vector<GpuTexture> textures;
    std::vector<GpuTriangle> triangles;
    std::vector<GpuSphere> spheres;
    std::vector<int> triangle_indices;
    std::vector<int> light_indices;
    std::vector<GpuDirectionalLight> directional_lights;
    std::vector<GpuBvhNode> bvh_nodes;
    std::vector<GpuMeshInstance> mesh_instances;
    std::vector<int> mesh_instance_indices;
    std::vector<GpuBvhNode> tlas_nodes;
};

struct GpuHit {
    float t = kInfinity;
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    Vec2 uv;
    int material = -1;
    int mesh = -1;
    int triangle = -1;
    int sphere = -1;
    Vec3 emission;
    bool front_face = true;
};

struct GpuMaterialSample {
    Vec3 direction;
    Vec3 weight;
    float pdf = 0.0f;
    bool delta = false;
};

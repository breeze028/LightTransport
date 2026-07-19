struct GpuMaterial {
    Vec3 albedo;
    int brdf_model = 0;
    float roughness = 0.5f;
    float metallic = 0.0f;
    Vec3 conductor_eta = {0.200438f, 0.924033f, 1.102212f};
    Vec3 conductor_k = {3.912949f, 2.452848f, 2.142188f};
    int texture_index = -1;
    int metallic_roughness_texture_index = -1;
    int roughness_texture_index = -1;
    int metallic_texture_index = -1;
    int specular_texture_index = -1;
    Vec2 texture_offset = {};
    Vec2 texture_scale = {1.0f, 1.0f};
    float texture_rotation = 0.0f;
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
    Vec2 emission_texture_offset = {};
    Vec2 emission_texture_scale = {1.0f, 1.0f};
    float emission_texture_rotation = 0.0f;
    Vec3 transmission_tint = {1.0f, 1.0f, 1.0f};
    float transmission = 0.0f;
    int transmission_texture_index = -1;
    float specular_ior = 1.5f;
    float specular_weight = 1.0f;
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

struct GpuPointLight {
    Vec3 position;
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
    Vec2 lightmap_uv0;
    Vec2 lightmap_uv1;
    Vec2 lightmap_uv2;
    int material = 0;
    int mesh = -1;
    int light_double_sided = 0;
    int has_lightmap = 0;
};

// Kept separate from GpuTriangle so candidate tests during BVH traversal do
// not fetch normal, UV, tangent, lightmap, and emission data before a hit.
struct GpuTraversalTriangle {
    Vec3 v0;
    Vec3 edge1;
    Vec3 edge2;
    int material_and_flags = 0;
};

constexpr int kTraversalMaterialAlphaBit = 0x40000000;
constexpr int kTraversalMaterialIndexMask = kTraversalMaterialAlphaBit - 1;

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

struct GpuTraversalBvhNode {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int left_or_first = -1;
    int right_or_neg_count = -1;
};

struct GpuTraversalBvh8Child {
    Vec3 bounds_min;
    int index = -1;
    Vec3 bounds_max;
    int count = 0;
};

struct GpuTraversalBvh8Node {
    GpuTraversalBvh8Child children[8];
    unsigned int child_octants = 0;
    int valid_mask = 0;
    int leaf_mask = 0;
};

enum class GpuTraversalLayout : int {
    Binary = 0,
    Bvh8 = 1,
    CwBvh = 2,
};

// 80-byte compressed wide BVH node in tinybvh BVH8_CWBVH layout.
// Stored as five float4 blocks so host conversion can copy tinybvh's output
// directly and CUDA traversal can follow the reference encoding.
struct GpuCwBvhNode {
    float4 block[5];
};

static_assert(sizeof(GpuCwBvhNode) == 80, "GpuCwBvhNode must stay 80 bytes");

struct GpuMeshInstance {
    Vec3 bounds_min;
    Vec3 bounds_max;
    int bvh_root = -1;
    int bvh8_root = -1;
    int cwbvh_root = -1;
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

struct GpuLightmap {
    int width = 0;
    int height = 0;
    Vec3* texels = nullptr;
};

struct GpuEnvironmentSampler {
    int width = 0;
    int height = 0;
    int texel_count = 0;
    float* pmf = nullptr;
    float* alias_probability = nullptr;
    int* alias_index = nullptr;
};

struct GpuLightSampler {
    int light_count = 0;
    float* pmf = nullptr;
    float* alias_probability = nullptr;
    int* alias_index = nullptr;
};

struct GpuScene {
    Camera camera;
    int material_count = 0;
    int triangle_count = 0;
    int sphere_count = 0;
    int bvh_node_count = 0;
    int bvh8_node_count = 0;
    int cwbvh_node_count = 0;
    int cwbvh_triangle_index_count = 0;
    int cwbvh_triangle_count = 0;
    int tlas_node_count = 0;
    int mesh_instance_count = 0;
    int use_two_level = 0;
    int light_count = 0;
    int directional_light_count = 0;
    int point_light_count = 0;
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
    GpuTraversalTriangle* traversal_triangles = nullptr;
    GpuSphere* spheres = nullptr;
    int* triangle_indices = nullptr;
    int* light_indices = nullptr;
    GpuDirectionalLight* directional_lights = nullptr;
    GpuPointLight* point_lights = nullptr;
    GpuBvhNode* bvh_nodes = nullptr;
    GpuTraversalBvhNode* traversal_bvh_nodes = nullptr;
    GpuTraversalBvh8Node* traversal_bvh8_nodes = nullptr;
    GpuCwBvhNode* traversal_cwbvh_nodes = nullptr;
    int* cwbvh_triangle_indices = nullptr;
    float4* cwbvh_triangles = nullptr;
    GpuMeshInstance* mesh_instances = nullptr;
    int* mesh_instance_indices = nullptr;
    GpuBvhNode* tlas_nodes = nullptr;
    GpuTraversalBvhNode* traversal_tlas_nodes = nullptr;
    GpuIrradianceVolume irradiance_volume;
    GpuLightmap lightmap;
    GpuEnvironmentSampler environment_sampler;
    GpuLightSampler restir_light_sampler;
};

struct PackedGpuScene {
    GpuScene scene;
    std::vector<GpuMaterial> materials;
    std::vector<GpuTexture> textures;
    std::vector<GpuTriangle> triangles;
    std::vector<GpuTraversalTriangle> traversal_triangles;
    std::vector<GpuSphere> spheres;
    std::vector<int> triangle_indices;
    std::vector<int> light_indices;
    std::vector<GpuDirectionalLight> directional_lights;
    std::vector<GpuPointLight> point_lights;
    std::vector<GpuBvhNode> bvh_nodes;
    std::vector<GpuTraversalBvhNode> traversal_bvh_nodes;
    std::vector<GpuTraversalBvh8Node> traversal_bvh8_nodes;
    std::vector<GpuCwBvhNode> traversal_cwbvh_nodes;
    std::vector<int> cwbvh_triangle_indices;
    std::vector<float4> cwbvh_triangles;
    std::vector<GpuMeshInstance> mesh_instances;
    std::vector<int> mesh_instance_indices;
    std::vector<GpuBvhNode> tlas_nodes;
    std::vector<GpuTraversalBvhNode> traversal_tlas_nodes;
};

struct GpuHit {
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
    int sphere = -1;
    Vec3 emission;
    bool has_lightmap = false;
    bool front_face = true;
};

struct GpuCompactHit {
    float t = kInfinity;
    float u = 0.0f;
    float v = 0.0f;
    int material = -1;
    int triangle = -1;
    int sphere = -1;
};

struct GpuMaterialSample {
    Vec3 direction;
    Vec3 weight;
    float pdf = 0.0f;
    bool delta = false;
};

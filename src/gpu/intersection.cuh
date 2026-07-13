__device__ bool intersect_triangle(const GpuTraversalTriangle& tri, const Ray& ray, float& t, float& u, float& v) {
    const Vec3 p = dcross(ray.direction, tri.edge2);
    const float det = ddot(tri.edge1, p);
    if (fabsf(det) < 1.0e-7f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 s = sub(ray.origin, tri.v0);
    u = inv_det * ddot(s, p);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3 q = dcross(s, tri.edge1);
    v = inv_det * ddot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    t = inv_det * ddot(tri.edge2, q);
    return t > 0.001f;
}

__device__ int traversal_material_index(int material_and_flags) {
    return material_and_flags & kTraversalMaterialIndexMask;
}

__device__ bool traversal_material_has_alpha(int material_and_flags) {
    return (material_and_flags & kTraversalMaterialAlphaBit) != 0;
}

__device__ Vec2 sphere_uv_gpu(Vec3 normal) {
    const float u = atan2f(normal.z, normal.x) / (2.0f * kPi) + 0.5f;
    const float v = acosf(dclamp(normal.y, -1.0f, 1.0f)) / kPi;
    return {u, 1.0f - v};
}

__device__ void sphere_basis_gpu(Vec3 normal, Vec3& tangent, Vec3& bitangent) {
    const Vec3 up = fabsf(normal.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    tangent = dnormalize(dcross(up, normal));
    bitangent = dcross(normal, tangent);
}

__device__ bool intersect_sphere_compact_gpu(const GpuSphere& sphere, int sphere_index, const Ray& ray, GpuCompactHit& hit) {
    const Vec3 oc = sub(ray.origin, sphere.center);
    const float a = ddot(ray.direction, ray.direction);
    const float half_b = ddot(oc, ray.direction);
    const float c = ddot(oc, oc) - sphere.radius * sphere.radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float root = sqrtf(discriminant);
    float t = (-half_b - root) / a;
    if (t <= 0.001f || t >= hit.t) {
        t = (-half_b + root) / a;
        if (t <= 0.001f || t >= hit.t) {
            return false;
        }
    }
    hit.t = t;
    hit.u = 0.0f;
    hit.v = 0.0f;
    hit.material = sphere.material;
    hit.triangle = -1;
    hit.sphere = sphere_index;
    return true;
}

__device__ bool intersect_sphere_gpu(const GpuSphere& sphere, const Ray& ray, GpuHit& hit) {
    const Vec3 oc = sub(ray.origin, sphere.center);
    const float a = ddot(ray.direction, ray.direction);
    const float half_b = ddot(oc, ray.direction);
    const float c = ddot(oc, oc) - sphere.radius * sphere.radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float root = sqrtf(discriminant);
    float t = (-half_b - root) / a;
    if (t <= 0.001f || t >= hit.t) {
        t = (-half_b + root) / a;
        if (t <= 0.001f || t >= hit.t) {
            return false;
        }
    }

    const Vec3 position = add(ray.origin, mul(ray.direction, t));
    const Vec3 outward_normal = dnormalize(divv(sub(position, sphere.center), sphere.radius));
    Vec3 tangent;
    Vec3 bitangent;
    sphere_basis_gpu(outward_normal, tangent, bitangent);
    hit.t = t;
    hit.position = position;
    hit.front_face = ddot(outward_normal, ray.direction) < 0.0f;
    hit.normal = hit.front_face ? outward_normal : mul(outward_normal, -1.0f);
    hit.tangent = tangent;
    hit.bitangent = bitangent;
    hit.uv = sphere_uv_gpu(outward_normal);
    hit.lightmap_uv = {};
    hit.material = sphere.material;
    hit.mesh = -1;
    hit.triangle = -1;
    hit.sphere = sphere.sphere;
    hit.has_lightmap = false;
    hit.emission = {};
    return true;
}

__device__ void fill_sphere_hit_gpu(const GpuSphere& sphere, const Ray& ray, float t, GpuHit& hit) {
    const Vec3 position = add(ray.origin, mul(ray.direction, t));
    const Vec3 outward_normal = dnormalize(divv(sub(position, sphere.center), sphere.radius));
    Vec3 tangent;
    Vec3 bitangent;
    sphere_basis_gpu(outward_normal, tangent, bitangent);
    hit.t = t;
    hit.position = position;
    hit.front_face = ddot(outward_normal, ray.direction) < 0.0f;
    hit.normal = hit.front_face ? outward_normal : mul(outward_normal, -1.0f);
    hit.tangent = tangent;
    hit.bitangent = bitangent;
    hit.uv = sphere_uv_gpu(outward_normal);
    hit.lightmap_uv = {};
    hit.material = sphere.material;
    hit.mesh = -1;
    hit.triangle = -1;
    hit.sphere = sphere.sphere;
    hit.has_lightmap = false;
    hit.emission = {};
}

__device__ bool intersect_spheres_gpu(const GpuScene& scene, const Ray& ray, GpuHit& hit) {
    bool found = false;
    for (int i = 0; i < scene.sphere_count; ++i) {
        const GpuSphere sphere = scene.spheres[i];
        if (sphere.material < 0 || sphere.material >= scene.material_count) {
            continue;
        }
        found = intersect_sphere_gpu(sphere, ray, hit) || found;
    }
    return found;
}

__device__ bool intersect_spheres_compact_gpu(const GpuScene& scene, const Ray& ray, GpuCompactHit& hit) {
    bool found = false;
    for (int i = 0; i < scene.sphere_count; ++i) {
        const GpuSphere sphere = scene.spheres[i];
        if (sphere.material < 0 || sphere.material >= scene.material_count) {
            continue;
        }
        found = intersect_sphere_compact_gpu(sphere, i, ray, hit) || found;
    }
    return found;
}

__device__ void fill_triangle_hit_gpu(const GpuScene& scene, const Ray& ray, int tri_index, float t, float u, float v, GpuHit& hit) {
    const GpuTriangle& tri = scene.triangles[tri_index];
    const Vec3 shading_normal = dnormalize(add(add(mul(tri.n0, 1.0f - u - v), mul(tri.n1, u)), mul(tri.n2, v)));
    hit.t = t;
    hit.position = add(ray.origin, mul(ray.direction, t));
    const Vec3 hit_normal = ddot(shading_normal, shading_normal) > 0.0f ? shading_normal : tri.normal;
    hit.front_face = ddot(tri.normal, ray.direction) < 0.0f;
    hit.normal = ddot(hit_normal, ray.direction) < 0.0f ? hit_normal : mul(hit_normal, -1.0f);
    hit.tangent = tri.tangent;
    hit.bitangent = tri.bitangent;
    hit.uv = add(add(mul(tri.uv0, 1.0f - u - v), mul(tri.uv1, u)), mul(tri.uv2, v));
    hit.lightmap_uv = add(add(mul(tri.lightmap_uv0, 1.0f - u - v), mul(tri.lightmap_uv1, u)), mul(tri.lightmap_uv2, v));
    hit.material = tri.material;
    hit.mesh = tri.mesh;
    hit.triangle = tri_index;
    hit.has_lightmap = tri.has_lightmap != 0;
    hit.emission = tri.emission;
}

__device__ bool fill_compact_hit_gpu(const GpuScene& scene, const Ray& ray, const GpuCompactHit& compact, GpuHit& hit) {
    if (compact.triangle >= 0 && compact.triangle < scene.triangle_count) {
        fill_triangle_hit_gpu(scene, ray, compact.triangle, compact.t, compact.u, compact.v, hit);
        return true;
    }
    if (compact.sphere >= 0 && compact.sphere < scene.sphere_count) {
        fill_sphere_hit_gpu(scene.spheres[compact.sphere], ray, compact.t, hit);
        return true;
    }
    return false;
}

__device__ Vec3 compact_hit_position_gpu(const Ray& ray, const GpuCompactHit& hit) {
    return add(ray.origin, mul(ray.direction, hit.t));
}

__device__ Vec2 compact_hit_uv_gpu(const GpuScene& scene, const Ray& ray, const GpuCompactHit& hit) {
    if (hit.triangle >= 0 && hit.triangle < scene.triangle_count) {
        const GpuTriangle& tri = scene.triangles[hit.triangle];
        return add(add(mul(tri.uv0, 1.0f - hit.u - hit.v), mul(tri.uv1, hit.u)), mul(tri.uv2, hit.v));
    }
    if (hit.sphere >= 0 && hit.sphere < scene.sphere_count) {
        const GpuSphere sphere = scene.spheres[hit.sphere];
        const Vec3 position = compact_hit_position_gpu(ray, hit);
        const Vec3 outward_normal = dnormalize(divv(sub(position, sphere.center), sphere.radius));
        return sphere_uv_gpu(outward_normal);
    }
    return {};
}

__device__ Vec3 inverse_ray_direction_gpu(Vec3 direction) {
    return {
        fabsf(direction.x) > 1.0e-8f ? 1.0f / direction.x : 0.0f,
        fabsf(direction.y) > 1.0e-8f ? 1.0f / direction.y : 0.0f,
        fabsf(direction.z) > 1.0e-8f ? 1.0f / direction.z : 0.0f,
    };
}

__device__ bool intersect_aabb_axis_gpu(float origin, float direction, float inv_direction, float lo, float hi, float& tmin, float& tmax) {
    if (fabsf(direction) <= 1.0e-8f) {
        return origin >= lo && origin <= hi;
    }
    float t0 = (lo - origin) * inv_direction;
    float t1 = (hi - origin) * inv_direction;
    if (inv_direction < 0.0f) {
        const float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    tmin = fmaxf(tmin, t0);
    tmax = fminf(tmax, t1);
    return tmax >= tmin;
}

__device__ bool intersect_aabb_gpu(Vec3 bounds_min, Vec3 bounds_max, const Ray& ray, Vec3 inv_direction, float max_t, float* out_tmin = nullptr) {
    float tmin = 0.001f;
    float tmax = max_t;
    if (!intersect_aabb_axis_gpu(ray.origin.x, ray.direction.x, inv_direction.x, bounds_min.x, bounds_max.x, tmin, tmax) ||
        !intersect_aabb_axis_gpu(ray.origin.y, ray.direction.y, inv_direction.y, bounds_min.y, bounds_max.y, tmin, tmax) ||
        !intersect_aabb_axis_gpu(ray.origin.z, ray.direction.z, inv_direction.z, bounds_min.z, bounds_max.z, tmin, tmax)) {
        return false;
    }
    if (out_tmin) {
        *out_tmin = tmin;
    }
    return true;
}

__device__ bool traversal_node_is_leaf(const GpuTraversalBvhNode& node) {
    return node.right_or_neg_count < 0;
}

__device__ int traversal_node_count(const GpuTraversalBvhNode& node) {
    return -node.right_or_neg_count;
}

__device__ bool intersect_bvh_gpu(const GpuScene& scene, const GpuBvhNode* nodes, int node_count, const int* indices, int index_count, const Ray& ray, Vec3 inv_direction, int root, GpuHit& hit) {
    bool found = false;
    int closest_triangle = -1;
    float closest_t = hit.t;
    float closest_u = 0.0f;
    float closest_v = 0.0f;
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= node_count) {
            continue;
        }
        const GpuBvhNode node = nodes[node_index];
        if (!intersect_aabb_gpu(node.bounds_min, node.bounds_max, ray, inv_direction, hit.t)) {
            continue;
        }

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= index_count) {
                    continue;
                }
                const int tri_index = indices[index_offset];
                if (tri_index < 0 || tri_index >= scene.triangle_count) {
                    continue;
                }
                const GpuTraversalTriangle& traversal_tri = scene.traversal_triangles[tri_index];
                const int material = traversal_material_index(traversal_tri.material_and_flags);
                if (material < 0 || material >= scene.material_count) {
                    continue;
                }
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(traversal_tri, ray, t, u, v) && t < hit.t) {
                    hit.t = t;
                    closest_t = t;
                    closest_u = u;
                    closest_v = v;
                    closest_triangle = tri_index;
                    found = true;
                }
            }
        } else {
            float left_t = kInfinity;
            float right_t = kInfinity;
            const bool hit_left = node.left >= 0 && node.left < node_count &&
                intersect_aabb_gpu(nodes[node.left].bounds_min, nodes[node.left].bounds_max, ray, inv_direction, hit.t, &left_t);
            const bool hit_right = node.right >= 0 && node.right < node_count &&
                intersect_aabb_gpu(nodes[node.right].bounds_min, nodes[node.right].bounds_max, ray, inv_direction, hit.t, &right_t);
            if (hit_left && hit_right) {
                const int near_child = left_t <= right_t ? node.left : node.right;
                const int far_child = left_t <= right_t ? node.right : node.left;
                if (stack_size < 64) stack[stack_size++] = far_child;
                if (stack_size < 64) stack[stack_size++] = near_child;
            } else if (hit_left) {
                if (stack_size < 64) stack[stack_size++] = node.left;
            } else if (hit_right) {
                if (stack_size < 64) stack[stack_size++] = node.right;
            }
        }
    }
    if (found) {
        fill_triangle_hit_gpu(scene, ray, closest_triangle, closest_t, closest_u, closest_v, hit);
    }
    return found;
}

__device__ bool intersect_bvh_compact_gpu(const GpuScene& scene, const GpuTraversalBvhNode* nodes, int node_count, const int* indices, int index_count, const Ray& ray, Vec3 inv_direction, int root, GpuCompactHit& hit) {
    bool found = false;
    int stack[64];
    int stack_size = 0;
    if (root < 0 || root >= node_count) {
        return false;
    }
    float root_t = 0.0f;
    if (!intersect_aabb_gpu(nodes[root].bounds_min, nodes[root].bounds_max, ray, inv_direction, hit.t, &root_t)) {
        return false;
    }
    stack[stack_size++] = root;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= node_count) {
            continue;
        }
        const GpuTraversalBvhNode node = nodes[node_index];

        if (traversal_node_is_leaf(node)) {
            const int first = node.left_or_first;
            const int count = traversal_node_count(node);
            for (int i = 0; i < count; ++i) {
                const int index_offset = first + i;
                if (index_offset < 0 || index_offset >= index_count) {
                    continue;
                }
                const int tri_index = indices[index_offset];
                if (tri_index < 0 || tri_index >= scene.triangle_count) {
                    continue;
                }
                const GpuTraversalTriangle& traversal_tri = scene.traversal_triangles[tri_index];
                const int material = traversal_material_index(traversal_tri.material_and_flags);
                if (material < 0 || material >= scene.material_count) {
                    continue;
                }
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(traversal_tri, ray, t, u, v) && t < hit.t) {
                    hit.t = t;
                    hit.u = u;
                    hit.v = v;
                    hit.material = traversal_tri.material_and_flags;
                    hit.triangle = tri_index;
                    hit.sphere = -1;
                    found = true;
                }
            }
        } else {
            float left_t = kInfinity;
            float right_t = kInfinity;
            const int left = node.left_or_first;
            const int right = node.right_or_neg_count;
            const bool hit_left = left >= 0 && left < node_count &&
                intersect_aabb_gpu(nodes[left].bounds_min, nodes[left].bounds_max, ray, inv_direction, hit.t, &left_t);
            const bool hit_right = right >= 0 && right < node_count &&
                intersect_aabb_gpu(nodes[right].bounds_min, nodes[right].bounds_max, ray, inv_direction, hit.t, &right_t);
            if (hit_left && hit_right) {
                const int near_child = left_t <= right_t ? left : right;
                const int far_child = left_t <= right_t ? right : left;
                const float near_t = left_t <= right_t ? left_t : right_t;
                const float far_t = left_t <= right_t ? right_t : left_t;
                if (far_t < hit.t && stack_size < 64) stack[stack_size++] = far_child;
                if (near_t < hit.t && stack_size < 64) stack[stack_size++] = near_child;
            } else if (hit_left) {
                if (left_t < hit.t && stack_size < 64) stack[stack_size++] = left;
            } else if (hit_right) {
                if (right_t < hit.t && stack_size < 64) stack[stack_size++] = right;
            }
        }
    }
    return found;
}

__device__ bool intersect_bvh8_leaf_compact_gpu(
    const GpuScene& scene,
    const int* indices,
    int index_count,
    const Ray& ray,
    int first,
    int count,
    GpuCompactHit& hit)
{
    bool found = false;
    #pragma unroll
    for (int i = 0; i < count; ++i) {
        const int index_offset = first + i;
        if (index_offset < 0 || index_offset >= index_count) {
            continue;
        }
        const int tri_index = indices[index_offset];
        if (tri_index < 0 || tri_index >= scene.triangle_count) {
            continue;
        }
        const GpuTraversalTriangle& traversal_tri = scene.traversal_triangles[tri_index];
        const int material = traversal_material_index(traversal_tri.material_and_flags);
        if (material < 0 || material >= scene.material_count) {
            continue;
        }
        float t = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (intersect_triangle(traversal_tri, ray, t, u, v) && t < hit.t) {
            hit.t = t;
            hit.u = u;
            hit.v = v;
            hit.material = traversal_tri.material_and_flags;
            hit.triangle = tri_index;
            hit.sphere = -1;
            found = true;
        }
    }
    return found;
}

__device__ __forceinline__ unsigned int ray_direction_octant_gpu(Vec3 direction) {
    unsigned int octant = 0;
    if (direction.x < 0.0f) octant |= 1u;
    if (direction.y < 0.0f) octant |= 2u;
    if (direction.z < 0.0f) octant |= 4u;
    return octant;
}

__device__ __forceinline__ unsigned int bvh8_child_priority_gpu(unsigned int child_octants, int slot, unsigned int ray_octant) {
    const unsigned int child_octant = (child_octants >> (slot * 3)) & 7u;
    return child_octant ^ ray_octant;
}

__device__ bool intersect_bvh8_compact_gpu(
    const GpuScene& scene,
    const GpuTraversalBvh8Node* nodes,
    int node_count,
    const int* indices,
    int index_count,
    const Ray& ray,
    Vec3 inv_direction,
    int root,
    GpuCompactHit& hit)
{
    if (root < 0 || root >= node_count) {
        return false;
    }

    bool found = false;
    const unsigned int ray_octant = ray_direction_octant_gpu(ray.direction);
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= node_count) {
            continue;
        }
        const GpuTraversalBvh8Node* node = &nodes[node_index];
        unsigned int internal_hit_mask = 0;
        const int valid_mask = node->valid_mask;
        const int leaf_mask = node->leaf_mask;
        const unsigned int child_octants = node->child_octants;

        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const unsigned int slot_mask = 1u << slot;
            if ((valid_mask & slot_mask) == 0) {
                continue;
            }
            const GpuTraversalBvh8Child& child = node->children[slot];
            if (!intersect_aabb_gpu(child.bounds_min, child.bounds_max, ray, inv_direction, hit.t)) {
                continue;
            }
            const int child_index = child.index;
            if ((leaf_mask & slot_mask) != 0) {
                found = intersect_bvh8_leaf_compact_gpu(
                    scene, indices, index_count, ray, child_index, child.count, hit) || found;
                continue;
            }
            if (child_index >= 0 && child_index < node_count) {
                internal_hit_mask |= slot_mask;
            }
        }

        for (int priority = 7; priority >= 0; --priority) {
            for (int slot = 7; slot >= 0; --slot) {
                const unsigned int slot_mask = 1u << slot;
                if ((internal_hit_mask & slot_mask) == 0 ||
                    bvh8_child_priority_gpu(child_octants, slot, ray_octant) != static_cast<unsigned int>(priority)) {
                    continue;
                }
                const int child_index = node->children[slot].index;
                if (stack_size < 64) {
                    stack[stack_size++] = child_index;
                }
            }
        }
    }
    return found;
}

__device__ bool intersect_blas_gpu(const GpuScene& scene, const Ray& ray, Vec3 inv_direction, int root, GpuHit& hit) {
    return intersect_bvh_gpu(scene, scene.bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, inv_direction, root, hit);
}

__device__ bool intersect_blas_compact_gpu(const GpuScene& scene, const Ray& ray, Vec3 inv_direction, int root, GpuCompactHit& hit) {
    return intersect_bvh_compact_gpu(scene, scene.traversal_bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, inv_direction, root, hit);
}

__device__ bool intersect_blas_bvh8_compact_gpu(const GpuScene& scene, const Ray& ray, Vec3 inv_direction, int root, GpuCompactHit& hit) {
    return intersect_bvh8_compact_gpu(scene, scene.traversal_bvh8_nodes, scene.bvh8_node_count, scene.triangle_indices, scene.triangle_count, ray, inv_direction, root, hit);
}

__device__ __forceinline__ signed char cwbvh_signed_byte_gpu(unsigned int value, int byte_index) {
    return static_cast<signed char>((value >> (byte_index * 8)) & 0xffu);
}

__device__ __forceinline__ unsigned int cwbvh_float_bits_gpu(float value) {
    return __float_as_uint(value);
}

__device__ __forceinline__ float cwbvh_pow2_scale_gpu(signed char exponent) {
    return __uint_as_float(static_cast<unsigned int>(static_cast<int>(exponent) + 127) << 23);
}

__device__ __forceinline__ unsigned int cwbvh_internal_mask_gpu(const GpuCwBvhNode& node) {
    return (cwbvh_float_bits_gpu(node.block[0].w) >> 24) & 0xffu;
}

__device__ __forceinline__ unsigned int cwbvh_child_base_gpu(const GpuCwBvhNode& node) {
    return cwbvh_float_bits_gpu(node.block[1].x);
}

__device__ __forceinline__ unsigned int cwbvh_triangle_base_gpu(const GpuCwBvhNode& node) {
    return cwbvh_float_bits_gpu(node.block[1].y);
}

__device__ __forceinline__ unsigned char cwbvh_node_byte_gpu(const GpuCwBvhNode& node, int byte_offset) {
    const float* values = reinterpret_cast<const float*>(node.block);
    const unsigned int word = cwbvh_float_bits_gpu(values[byte_offset >> 2]);
    return static_cast<unsigned char>((word >> ((byte_offset & 3) * 8)) & 0xffu);
}

__device__ __forceinline__ bool intersect_cwbvh_quantized_child_gpu(
    const GpuCwBvhNode& node,
    int slot,
    float orig_x,
    float orig_y,
    float orig_z,
    float idir_x,
    float idir_y,
    float idir_z,
    bool neg_x,
    bool neg_y,
    bool neg_z,
    float max_t)
{
    const unsigned char qlo_x = cwbvh_node_byte_gpu(node, 32 + slot);
    const unsigned char qlo_y = cwbvh_node_byte_gpu(node, 40 + slot);
    const unsigned char qlo_z = cwbvh_node_byte_gpu(node, 48 + slot);
    const unsigned char qhi_x = cwbvh_node_byte_gpu(node, 56 + slot);
    const unsigned char qhi_y = cwbvh_node_byte_gpu(node, 64 + slot);
    const unsigned char qhi_z = cwbvh_node_byte_gpu(node, 72 + slot);
    const unsigned char qmin_x = neg_x ? qhi_x : qlo_x;
    const unsigned char qmax_x = neg_x ? qlo_x : qhi_x;
    const unsigned char qmin_y = neg_y ? qhi_y : qlo_y;
    const unsigned char qmax_y = neg_y ? qlo_y : qhi_y;
    const unsigned char qmin_z = neg_z ? qhi_z : qlo_z;
    const unsigned char qmax_z = neg_z ? qlo_z : qhi_z;

    const float tx0 = fmaf(static_cast<float>(qmin_x), idir_x, orig_x);
    const float tx1 = fmaf(static_cast<float>(qmax_x), idir_x, orig_x);
    const float ty0 = fmaf(static_cast<float>(qmin_y), idir_y, orig_y);
    const float ty1 = fmaf(static_cast<float>(qmax_y), idir_y, orig_y);
    const float tz0 = fmaf(static_cast<float>(qmin_z), idir_z, orig_z);
    const float tz1 = fmaf(static_cast<float>(qmax_z), idir_z, orig_z);

    const float tmin = fmaxf(fmaxf(fmaxf(tx0, ty0), tz0), 0.001f);
    const float tmax = fminf(fminf(fminf(tx1, ty1), tz1), max_t);
    return tmax >= tmin;
}

__device__ __forceinline__ bool intersect_cwbvh_triangle_compact_gpu(
    const GpuScene& scene,
    const Ray& ray,
    int tri_addr,
    GpuCompactHit& hit)
{
    if (tri_addr < 0 || tri_addr + 2 >= scene.cwbvh_triangle_count || scene.cwbvh_triangles == nullptr) {
        return false;
    }
    const float4 edge2_block = scene.cwbvh_triangles[tri_addr + 0];
    const float4 edge1_block = scene.cwbvh_triangles[tri_addr + 1];
    const float4 v0_block = scene.cwbvh_triangles[tri_addr + 2];
    const int tri_index = static_cast<int>(cwbvh_float_bits_gpu(v0_block.w));
    if (tri_index < 0 || tri_index >= scene.triangle_count) {
        return false;
    }
    const int material_and_flags = static_cast<int>(cwbvh_float_bits_gpu(edge2_block.w));
    const GpuTraversalTriangle traversal_tri{
        Vec3{v0_block.x, v0_block.y, v0_block.z},
        Vec3{edge1_block.x, edge1_block.y, edge1_block.z},
        Vec3{edge2_block.x, edge2_block.y, edge2_block.z},
        material_and_flags,
    };
    float t = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    if (intersect_triangle(traversal_tri, ray, t, u, v) && t < hit.t) {
        hit.t = t;
        hit.u = u;
        hit.v = v;
        hit.material = material_and_flags;
        hit.triangle = tri_index;
        hit.sphere = -1;
        return true;
    }
    return false;
}

__device__ bool intersect_cwbvh_compact_gpu(
    const GpuScene& scene,
    const Ray& ray,
    Vec3 inv_direction,
    int root,
    GpuCompactHit& hit)
{
    if (root < 0 || root >= scene.cwbvh_node_count || scene.traversal_cwbvh_nodes == nullptr ||
        scene.cwbvh_triangles == nullptr) {
        return false;
    }

    bool found = false;
    uint2 stack[32];
    int stack_size = 0;
    const unsigned int octinv = 7u - ((ray.direction.x < 0.0f ? 4u : 0u) |
        (ray.direction.y < 0.0f ? 2u : 0u) |
        (ray.direction.z < 0.0f ? 1u : 0u));
    uint2 ngroup = make_uint2(static_cast<unsigned int>(root), 0x80000000u);
    uint2 tgroup = make_uint2(0u, 0u);

    while (true) {
        if (ngroup.y > 0x00ffffffu) {
            const unsigned int hits = ngroup.y;
            const unsigned int imask = ngroup.y;
            const unsigned int child_bit_index = 31u - static_cast<unsigned int>(__clz(hits));
            const unsigned int child_node_base = ngroup.x;
            ngroup.y &= ~(1u << child_bit_index);
            if (ngroup.y > 0x00ffffffu && stack_size < 32) {
                stack[stack_size++] = ngroup;
            }

            const unsigned int slot_index = (child_bit_index - 24u) ^ octinv;
            const unsigned int lower_slot_mask = slot_index == 0u ? 0u : ((1u << slot_index) - 1u);
            const unsigned int relative_index = __popc(imask & lower_slot_mask);
            const int node_index = static_cast<int>(child_node_base + relative_index);
            if (node_index < 0 || node_index >= scene.cwbvh_node_count) {
                ngroup = make_uint2(0u, 0u);
                tgroup = make_uint2(0u, 0u);
            } else {
                const GpuCwBvhNode& node = scene.traversal_cwbvh_nodes[node_index];
                const unsigned int node_imask = cwbvh_internal_mask_gpu(node);
                const unsigned int exyz_imask = cwbvh_float_bits_gpu(node.block[0].w);
                const float sx = cwbvh_pow2_scale_gpu(cwbvh_signed_byte_gpu(exyz_imask, 0));
                const float sy = cwbvh_pow2_scale_gpu(cwbvh_signed_byte_gpu(exyz_imask, 1));
                const float sz = cwbvh_pow2_scale_gpu(cwbvh_signed_byte_gpu(exyz_imask, 2));
                const float orig_x = (node.block[0].x - ray.origin.x) * inv_direction.x;
                const float orig_y = (node.block[0].y - ray.origin.y) * inv_direction.y;
                const float orig_z = (node.block[0].z - ray.origin.z) * inv_direction.z;
                const float idir_x = sx * inv_direction.x;
                const float idir_y = sy * inv_direction.y;
                const float idir_z = sz * inv_direction.z;
                const bool neg_x = inv_direction.x < 0.0f;
                const bool neg_y = inv_direction.y < 0.0f;
                const bool neg_z = inv_direction.z < 0.0f;
                const unsigned int meta0 = cwbvh_float_bits_gpu(node.block[1].z);
                const unsigned int meta1 = cwbvh_float_bits_gpu(node.block[1].w);

                unsigned int hitmask = 0;
                #pragma unroll
                for (int slot = 0; slot < 8; ++slot) {
                    const unsigned int meta_word = slot < 4 ? meta0 : meta1;
                    const unsigned char meta = static_cast<unsigned char>((meta_word >> ((slot & 3) * 8)) & 0xffu);
                    if (meta == 0) {
                        continue;
                    }
                    if (!intersect_cwbvh_quantized_child_gpu(node, slot, orig_x, orig_y, orig_z, idir_x, idir_y, idir_z,
                            neg_x, neg_y, neg_z, hit.t)) {
                        continue;
                    }
                    if ((node_imask & (1u << slot)) != 0) {
                        hitmask |= 1u << (24u + (static_cast<unsigned int>(slot) ^ octinv));
                    } else {
                        hitmask |= static_cast<unsigned int>(meta >> 5) << (meta & 31);
                    }
                }
                ngroup = make_uint2(cwbvh_child_base_gpu(node), (hitmask & 0xff000000u) | node_imask);
                tgroup = make_uint2(cwbvh_triangle_base_gpu(node), hitmask & 0x00ffffffu);
            }
        } else {
            tgroup = ngroup;
            ngroup = make_uint2(0u, 0u);
        }

        while (tgroup.y != 0u) {
            const unsigned int tri_bit = 31u - static_cast<unsigned int>(__clz(tgroup.y));
            tgroup.y &= ~(1u << tri_bit);
            found = intersect_cwbvh_triangle_compact_gpu(scene, ray, static_cast<int>(tgroup.x + tri_bit * 3u), hit) || found;
        }

        if (ngroup.y <= 0x00ffffffu) {
            if (stack_size > 0) {
                ngroup = stack[--stack_size];
            } else {
                break;
            }
        }
    }
    return found;
}

__device__ bool intersect_blas_cwbvh_compact_gpu(const GpuScene& scene, const Ray& ray, Vec3 inv_direction, int root, GpuCompactHit& hit) {
    return intersect_cwbvh_compact_gpu(scene, ray, inv_direction, root, hit);
}

__device__ bool intersect_gpu(const GpuScene& scene, const Ray& ray, GpuHit& hit) {
    const Vec3 inv_direction = inverse_ray_direction_gpu(ray.direction);
    if (scene.use_two_level == 0) {
        bool found = scene.bvh_node_count > 0 &&
            intersect_bvh_gpu(scene, scene.bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, inv_direction, 0, hit);
        found = intersect_spheres_gpu(scene, ray, hit) || found;
        return found;
    }

    bool found = false;
    if (scene.tlas_node_count > 0 && scene.mesh_instance_count > 0) {
        int stack[64];
        int stack_size = 0;
        stack[stack_size++] = 0;

        while (stack_size > 0) {
            const int node_index = stack[--stack_size];
            if (node_index < 0 || node_index >= scene.tlas_node_count) {
                continue;
            }
            const GpuBvhNode node = scene.tlas_nodes[node_index];
            if (!intersect_aabb_gpu(node.bounds_min, node.bounds_max, ray, inv_direction, hit.t)) {
                continue;
            }

            if (node.count > 0) {
                for (int i = 0; i < node.count; ++i) {
                    const int index_offset = node.first + i;
                    if (index_offset < 0 || index_offset >= scene.mesh_instance_count) {
                        continue;
                    }
                    const int instance_index = scene.mesh_instance_indices[index_offset];
                    if (instance_index < 0 || instance_index >= scene.mesh_instance_count) {
                        continue;
                    }
                    const GpuMeshInstance instance = scene.mesh_instances[instance_index];
                    if (instance.bvh_root >= 0 &&
                        intersect_aabb_gpu(instance.bounds_min, instance.bounds_max, ray, inv_direction, hit.t)) {
                        found = intersect_blas_gpu(scene, ray, inv_direction, instance.bvh_root, hit) || found;
                    }
                }
            } else {
                float left_t = kInfinity;
                float right_t = kInfinity;
                const bool hit_left = node.left >= 0 && node.left < scene.tlas_node_count &&
                    intersect_aabb_gpu(scene.tlas_nodes[node.left].bounds_min, scene.tlas_nodes[node.left].bounds_max, ray, inv_direction, hit.t, &left_t);
                const bool hit_right = node.right >= 0 && node.right < scene.tlas_node_count &&
                    intersect_aabb_gpu(scene.tlas_nodes[node.right].bounds_min, scene.tlas_nodes[node.right].bounds_max, ray, inv_direction, hit.t, &right_t);
                if (hit_left && hit_right) {
                    const int near_child = left_t <= right_t ? node.left : node.right;
                    const int far_child = left_t <= right_t ? node.right : node.left;
                    if (stack_size < 64) stack[stack_size++] = far_child;
                    if (stack_size < 64) stack[stack_size++] = near_child;
                } else if (hit_left) {
                    if (stack_size < 64) stack[stack_size++] = node.left;
                } else if (hit_right) {
                    if (stack_size < 64) stack[stack_size++] = node.right;
                }
            }
        }
    }
    found = intersect_spheres_gpu(scene, ray, hit) || found;
    return found;
}

template <bool TwoLevel, GpuTraversalLayout Layout>
__device__ bool intersect_compact_gpu(const GpuScene& scene, const Ray& ray, GpuCompactHit& hit) {
    const Vec3 inv_direction = inverse_ray_direction_gpu(ray.direction);
    if constexpr (!TwoLevel) {
        bool found = false;
        if constexpr (Layout == GpuTraversalLayout::Bvh8) {
            found = scene.bvh8_node_count > 0 &&
                intersect_bvh8_compact_gpu(scene, scene.traversal_bvh8_nodes, scene.bvh8_node_count, scene.triangle_indices, scene.triangle_count, ray, inv_direction, 0, hit);
        } else {
            found = scene.bvh_node_count > 0 &&
                intersect_bvh_compact_gpu(scene, scene.traversal_bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, inv_direction, 0, hit);
        }
        found = intersect_spheres_compact_gpu(scene, ray, hit) || found;
        return found;
    }

    bool found = false;
    if constexpr (Layout != GpuTraversalLayout::Binary) {
        if (scene.tlas_node_count > 0 && scene.mesh_instance_count > 0) {
            int stack[64];
            int stack_size = 0;
            float root_t = 0.0f;
            if (intersect_aabb_gpu(scene.traversal_tlas_nodes[0].bounds_min, scene.traversal_tlas_nodes[0].bounds_max, ray, inv_direction, hit.t, &root_t)) {
                stack[stack_size++] = 0;
            }

            while (stack_size > 0) {
                const int node_index = stack[--stack_size];
                if (node_index < 0 || node_index >= scene.tlas_node_count) {
                    continue;
                }
                const GpuTraversalBvhNode node = scene.traversal_tlas_nodes[node_index];

                if (traversal_node_is_leaf(node)) {
                    const int first = node.left_or_first;
                    const int count = traversal_node_count(node);
                    for (int i = 0; i < count; ++i) {
                        const int index_offset = first + i;
                        if (index_offset < 0 || index_offset >= scene.mesh_instance_count) {
                            continue;
                        }
                        const int instance_index = scene.mesh_instance_indices[index_offset];
                        if (instance_index < 0 || instance_index >= scene.mesh_instance_count) {
                            continue;
                        }
                        const GpuMeshInstance instance = scene.mesh_instances[instance_index];
                        if (intersect_aabb_gpu(instance.bounds_min, instance.bounds_max, ray, inv_direction, hit.t)) {
                            if constexpr (Layout == GpuTraversalLayout::CwBvh) {
                                if (instance.cwbvh_root >= 0) {
                                    found = intersect_blas_cwbvh_compact_gpu(scene, ray, inv_direction, instance.cwbvh_root, hit) || found;
                                } else if (instance.bvh8_root >= 0) {
                                    found = intersect_blas_bvh8_compact_gpu(scene, ray, inv_direction, instance.bvh8_root, hit) || found;
                                } else if (instance.bvh_root >= 0) {
                                    found = intersect_blas_compact_gpu(scene, ray, inv_direction, instance.bvh_root, hit) || found;
                                }
                            } else if (instance.bvh8_root >= 0) {
                                found = intersect_blas_bvh8_compact_gpu(scene, ray, inv_direction, instance.bvh8_root, hit) || found;
                            } else if (instance.bvh_root >= 0) {
                                found = intersect_blas_compact_gpu(scene, ray, inv_direction, instance.bvh_root, hit) || found;
                            }
                        }
                    }
                } else {
                    float left_t = kInfinity;
                    float right_t = kInfinity;
                    const int left = node.left_or_first;
                    const int right = node.right_or_neg_count;
                    const bool hit_left = left >= 0 && left < scene.tlas_node_count &&
                        intersect_aabb_gpu(scene.traversal_tlas_nodes[left].bounds_min, scene.traversal_tlas_nodes[left].bounds_max, ray, inv_direction, hit.t, &left_t);
                    const bool hit_right = right >= 0 && right < scene.tlas_node_count &&
                        intersect_aabb_gpu(scene.traversal_tlas_nodes[right].bounds_min, scene.traversal_tlas_nodes[right].bounds_max, ray, inv_direction, hit.t, &right_t);
                    if (hit_left && hit_right) {
                        const int near_child = left_t <= right_t ? left : right;
                        const int far_child = left_t <= right_t ? right : left;
                        const float near_t = left_t <= right_t ? left_t : right_t;
                        const float far_t = left_t <= right_t ? right_t : left_t;
                        if (far_t < hit.t && stack_size < 64) stack[stack_size++] = far_child;
                        if (near_t < hit.t && stack_size < 64) stack[stack_size++] = near_child;
                    } else if (hit_left) {
                        if (left_t < hit.t && stack_size < 64) stack[stack_size++] = left;
                    } else if (hit_right) {
                        if (right_t < hit.t && stack_size < 64) stack[stack_size++] = right;
                    }
                }
            }
        }
        found = intersect_spheres_compact_gpu(scene, ray, hit) || found;
        return found;
    }

    if (scene.tlas_node_count > 0 && scene.mesh_instance_count > 0) {
        int stack[64];
        int stack_size = 0;
        float root_t = 0.0f;
        if (intersect_aabb_gpu(scene.traversal_tlas_nodes[0].bounds_min, scene.traversal_tlas_nodes[0].bounds_max, ray, inv_direction, hit.t, &root_t)) {
            stack[stack_size++] = 0;
        }

        while (stack_size > 0) {
            const int node_index = stack[--stack_size];
            if (node_index < 0 || node_index >= scene.tlas_node_count) {
                continue;
            }
            const GpuTraversalBvhNode node = scene.traversal_tlas_nodes[node_index];

            if (traversal_node_is_leaf(node)) {
                const int first = node.left_or_first;
                const int count = traversal_node_count(node);
                for (int i = 0; i < count; ++i) {
                    const int index_offset = first + i;
                    if (index_offset < 0 || index_offset >= scene.mesh_instance_count) {
                        continue;
                    }
                    const int instance_index = scene.mesh_instance_indices[index_offset];
                    if (instance_index < 0 || instance_index >= scene.mesh_instance_count) {
                        continue;
                    }
                    const GpuMeshInstance instance = scene.mesh_instances[instance_index];
                    if (instance.bvh_root >= 0 &&
                        intersect_aabb_gpu(instance.bounds_min, instance.bounds_max, ray, inv_direction, hit.t)) {
                        found = intersect_blas_compact_gpu(scene, ray, inv_direction, instance.bvh_root, hit) || found;
                    }
                }
            } else {
                float left_t = kInfinity;
                float right_t = kInfinity;
                const int left = node.left_or_first;
                const int right = node.right_or_neg_count;
                const bool hit_left = left >= 0 && left < scene.tlas_node_count &&
                    intersect_aabb_gpu(scene.traversal_tlas_nodes[left].bounds_min, scene.traversal_tlas_nodes[left].bounds_max, ray, inv_direction, hit.t, &left_t);
                const bool hit_right = right >= 0 && right < scene.tlas_node_count &&
                    intersect_aabb_gpu(scene.traversal_tlas_nodes[right].bounds_min, scene.traversal_tlas_nodes[right].bounds_max, ray, inv_direction, hit.t, &right_t);
                if (hit_left && hit_right) {
                    const int near_child = left_t <= right_t ? left : right;
                    const int far_child = left_t <= right_t ? right : left;
                    const float near_t = left_t <= right_t ? left_t : right_t;
                    const float far_t = left_t <= right_t ? right_t : left_t;
                    if (far_t < hit.t && stack_size < 64) stack[stack_size++] = far_child;
                    if (near_t < hit.t && stack_size < 64) stack[stack_size++] = near_child;
                } else if (hit_left) {
                    if (left_t < hit.t && stack_size < 64) stack[stack_size++] = left;
                } else if (hit_right) {
                    if (right_t < hit.t && stack_size < 64) stack[stack_size++] = right;
                }
            }
        }
    }
    found = intersect_spheres_compact_gpu(scene, ray, hit) || found;
    return found;
}

__device__ bool intersect_triangle(const GpuTriangle& tri, const Ray& ray, float& t, float& u, float& v) {
    const Vec3 edge1 = sub(tri.v1, tri.v0);
    const Vec3 edge2 = sub(tri.v2, tri.v0);
    const Vec3 p = dcross(ray.direction, edge2);
    const float det = ddot(edge1, p);
    if (fabsf(det) < 1.0e-7f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 s = sub(ray.origin, tri.v0);
    u = inv_det * ddot(s, p);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3 q = dcross(s, edge1);
    v = inv_det * ddot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    t = inv_det * ddot(edge2, q);
    return t > 0.001f;
}

__device__ bool intersect_aabb_gpu(Vec3 bounds_min, Vec3 bounds_max, const Ray& ray, float max_t, float* out_tmin = nullptr) {
    float tmin = 0.001f;
    float tmax = max_t;
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float direction = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float lo = axis == 0 ? bounds_min.x : axis == 1 ? bounds_min.y : bounds_min.z;
        const float hi = axis == 0 ? bounds_max.x : axis == 1 ? bounds_max.y : bounds_max.z;
        if (fabsf(direction) < 1.0e-8f) {
            if (origin < lo || origin > hi) {
                return false;
            }
            continue;
        }
        const float inv = 1.0f / direction;
        float t0 = (lo - origin) * inv;
        float t1 = (hi - origin) * inv;
        if (inv < 0.0f) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
        if (tmax < tmin) {
            return false;
        }
    }
    if (out_tmin) {
        *out_tmin = tmin;
    }
    return true;
}

__device__ bool intersect_bvh_gpu(const GpuScene& scene, const GpuBvhNode* nodes, int node_count, const int* indices, int index_count, const Ray& ray, int root, GpuHit& hit) {
    bool found = false;
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= node_count) {
            continue;
        }
        const GpuBvhNode node = nodes[node_index];
        if (!intersect_aabb_gpu(node.bounds_min, node.bounds_max, ray, hit.t)) {
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
                const GpuTriangle& tri = scene.triangles[tri_index];
                if (tri.material < 0 || tri.material >= scene.material_count) {
                    continue;
                }
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(tri, ray, t, u, v) && t < hit.t) {
                    const Vec3 shading_normal = dnormalize(add(add(mul(tri.n0, 1.0f - u - v), mul(tri.n1, u)), mul(tri.n2, v)));
                    hit.t = t;
                    hit.position = add(ray.origin, mul(ray.direction, t));
                    const Vec3 hit_normal = ddot(shading_normal, shading_normal) > 0.0f ? shading_normal : tri.normal;
                    hit.front_face = ddot(tri.normal, ray.direction) < 0.0f;
                    hit.normal = ddot(hit_normal, ray.direction) < 0.0f ? hit_normal : mul(hit_normal, -1.0f);
                    hit.tangent = tri.tangent;
                    hit.bitangent = tri.bitangent;
                    hit.uv = add(add(mul(tri.uv0, 1.0f - u - v), mul(tri.uv1, u)), mul(tri.uv2, v));
                    hit.material = tri.material;
                    hit.mesh = tri.mesh;
                    hit.triangle = tri_index;
                    hit.emission = tri.emission;
                    found = true;
                }
            }
        } else {
            float left_t = kInfinity;
            float right_t = kInfinity;
            const bool hit_left = node.left >= 0 && node.left < node_count &&
                intersect_aabb_gpu(nodes[node.left].bounds_min, nodes[node.left].bounds_max, ray, hit.t, &left_t);
            const bool hit_right = node.right >= 0 && node.right < node_count &&
                intersect_aabb_gpu(nodes[node.right].bounds_min, nodes[node.right].bounds_max, ray, hit.t, &right_t);
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
    return found;
}

__device__ bool intersect_blas_gpu(const GpuScene& scene, const Ray& ray, int root, GpuHit& hit) {
    return intersect_bvh_gpu(scene, scene.bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, root, hit);
}

__device__ bool intersect_gpu(const GpuScene& scene, const Ray& ray, GpuHit& hit) {
    if (scene.use_two_level == 0) {
        return scene.bvh_node_count > 0 &&
            intersect_bvh_gpu(scene, scene.bvh_nodes, scene.bvh_node_count, scene.triangle_indices, scene.triangle_count, ray, 0, hit);
    }
    if (scene.tlas_node_count <= 0 || scene.mesh_instance_count <= 0) {
        return false;
    }

    bool found = false;
    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = 0;

    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || node_index >= scene.tlas_node_count) {
            continue;
        }
        const GpuBvhNode node = scene.tlas_nodes[node_index];
        if (!intersect_aabb_gpu(node.bounds_min, node.bounds_max, ray, hit.t)) {
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
                    intersect_aabb_gpu(instance.bounds_min, instance.bounds_max, ray, hit.t)) {
                    found = intersect_blas_gpu(scene, ray, instance.bvh_root, hit) || found;
                }
            }
        } else {
            float left_t = kInfinity;
            float right_t = kInfinity;
            const bool hit_left = node.left >= 0 && node.left < scene.tlas_node_count &&
                intersect_aabb_gpu(scene.tlas_nodes[node.left].bounds_min, scene.tlas_nodes[node.left].bounds_max, ray, hit.t, &left_t);
            const bool hit_right = node.right >= 0 && node.right < scene.tlas_node_count &&
                intersect_aabb_gpu(scene.tlas_nodes[node.right].bounds_min, scene.tlas_nodes[node.right].bounds_max, ray, hit.t, &right_t);
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
    return found;
}


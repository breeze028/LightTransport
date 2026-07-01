bool intersect_aabb(const Aabb& bounds, const Ray& ray, float max_t) {
    float tmin = 0.001f;
    float tmax = max_t;
    for (int axis = 0; axis < 3; ++axis) {
        const float origin = axis == 0 ? ray.origin.x : axis == 1 ? ray.origin.y : ray.origin.z;
        const float direction = axis == 0 ? ray.direction.x : axis == 1 ? ray.direction.y : ray.direction.z;
        const float lo = axis == 0 ? bounds.min.x : axis == 1 ? bounds.min.y : bounds.min.z;
        const float hi = axis == 0 ? bounds.max.x : axis == 1 ? bounds.max.y : bounds.max.z;
        if (std::fabs(direction) < 1.0e-8f) {
            if (origin < lo || origin > hi) {
                return false;
            }
            continue;
        }
        const float inv = 1.0f / direction;
        float t0 = (lo - origin) * inv;
        float t1 = (hi - origin) * inv;
        if (inv < 0.0f) {
            std::swap(t0, t1);
        }
        tmin = std::max(tmin, t0);
        tmax = std::min(tmax, t1);
        if (tmax < tmin) {
            return false;
        }
    }
    return true;
}

bool intersect_triangle(const Triangle& tri, const Ray& ray, float& t, float& u, float& v) {
    const Vec3 edge1 = tri.v1 - tri.v0;
    const Vec3 edge2 = tri.v2 - tri.v0;
    const Vec3 p = cross(ray.direction, edge2);
    const float det = dot(edge1, p);
    if (std::fabs(det) < 1.0e-7f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 s = ray.origin - tri.v0;
    u = inv_det * dot(s, p);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vec3 q = cross(s, edge1);
    v = inv_det * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    t = inv_det * dot(edge2, q);
    return t > 0.001f;
}

Vec2 sphere_uv(Vec3 normal) {
    const float u = std::atan2(normal.z, normal.x) / (2.0f * kPi) + 0.5f;
    const float v = std::acos(std::clamp(normal.y, -1.0f, 1.0f)) / kPi;
    return {u, 1.0f - v};
}

void sphere_basis(Vec3 normal, Vec3& tangent, Vec3& bitangent) {
    const Vec3 up = std::fabs(normal.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    tangent = normalize(cross(up, normal));
    bitangent = cross(normal, tangent);
}

bool intersect_sphere(const RenderSphere& sphere, const Ray& ray, Hit& hit) {
    const Vec3 oc = ray.origin - sphere.center;
    const float a = dot(ray.direction, ray.direction);
    const float half_b = dot(oc, ray.direction);
    const float c = dot(oc, oc) - sphere.radius * sphere.radius;
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

    const Vec3 position = ray.origin + ray.direction * t;
    const Vec3 outward_normal = normalize((position - sphere.center) / sphere.radius);
    Vec3 tangent;
    Vec3 bitangent;
    sphere_basis(outward_normal, tangent, bitangent);
    hit.t = t;
    hit.position = position;
    hit.front_face = dot(outward_normal, ray.direction) < 0.0f;
    hit.normal = face_forward(outward_normal, ray.direction);
    hit.tangent = tangent;
    hit.bitangent = bitangent;
    hit.uv = sphere_uv(outward_normal);
    hit.lightmap_uv = {};
    hit.material = sphere.material;
    hit.mesh = -1;
    hit.triangle = -1;
    hit.has_lightmap = false;
    return true;
}

bool intersect_spheres(const RenderScene& render_scene, const Ray& ray, Hit& hit) {
    bool found = false;
    for (const RenderSphere& sphere : render_scene.spheres) {
        found = intersect_sphere(sphere, ray, hit) || found;
    }
    return found;
}

bool intersect_bvh_nodes(const RenderScene& render_scene, const std::vector<BvhNode>& nodes, const std::vector<int>& indices, const Ray& ray, int root, Hit& hit) {
    bool found = false;
    int stack[kTraversalStackSize] = {};
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        const BvhNode& node = nodes[static_cast<size_t>(stack[--stack_size])];
        if (!intersect_aabb(node.bounds, ray, hit.t)) {
            continue;
        }
        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                const int index_offset = node.first + i;
                if (index_offset < 0 || index_offset >= static_cast<int>(indices.size())) {
                    continue;
                }
                const int tri_index = indices[static_cast<size_t>(index_offset)];
                if (tri_index < 0 || tri_index >= static_cast<int>(render_scene.triangles.size())) {
                    continue;
                }
                const Triangle& tri = render_scene.triangles[static_cast<size_t>(tri_index)];
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (intersect_triangle(tri, ray, t, u, v) && t < hit.t) {
                    const Vec3 shading_normal = normalize(tri.n0 * (1.0f - u - v) + tri.n1 * u + tri.n2 * v);
                    hit.t = t;
                    hit.position = ray.origin + ray.direction * t;
                    hit.front_face = dot(tri.normal, ray.direction) < 0.0f;
                    hit.normal = face_forward(dot(shading_normal, shading_normal) > 0.0f ? shading_normal : tri.normal, ray.direction);
                    hit.tangent = tri.tangent;
                    hit.bitangent = tri.bitangent;
                    hit.uv = tri.uv0 * (1.0f - u - v) + tri.uv1 * u + tri.uv2 * v;
                    hit.lightmap_uv = tri.lightmap_uv0 * (1.0f - u - v) + tri.lightmap_uv1 * u + tri.lightmap_uv2 * v;
                    hit.material = tri.material;
                    hit.mesh = tri.mesh;
                    hit.triangle = tri_index;
                    hit.has_lightmap = tri.has_lightmap;
                    found = true;
                }
            }
        } else {
            if (node.left >= 0 && node.left < static_cast<int>(nodes.size()) && stack_size < kTraversalStackSize) {
                stack[stack_size++] = node.left;
            }
            if (node.right >= 0 && node.right < static_cast<int>(nodes.size()) && stack_size < kTraversalStackSize) {
                stack[stack_size++] = node.right;
            }
        }
    }
    return found;
}

bool intersect_blas(const RenderScene& render_scene, const Ray& ray, int root, Hit& hit) {
    return intersect_bvh_nodes(render_scene, render_scene.bvh_nodes, render_scene.triangle_indices, ray, root, hit);
}

bool use_two_level(const RenderScene& render_scene, AccelerationStructure acceleration_structure) {
    if (acceleration_structure == AccelerationStructure::TwoLevel) {
        return true;
    }
    if (acceleration_structure == AccelerationStructure::Flat) {
        return false;
    }
    return render_scene.mesh_instances.size() > 1;
}

bool intersect_scene(const RenderScene& render_scene, const Ray& ray, Hit& hit, AccelerationStructure acceleration_structure) {
    if (!use_two_level(render_scene, acceleration_structure)) {
        bool found = !render_scene.flat_bvh_nodes.empty() &&
            intersect_bvh_nodes(render_scene, render_scene.flat_bvh_nodes, render_scene.flat_triangle_indices, ray, 0, hit);
        found = intersect_spheres(render_scene, ray, hit) || found;
        return found;
    }

    bool found = false;
    if (!render_scene.tlas_nodes.empty() && !render_scene.mesh_instances.empty()) {
        int stack[kTraversalStackSize] = {};
        int stack_size = 0;
        stack[stack_size++] = 0;

        while (stack_size > 0) {
            const BvhNode& node = render_scene.tlas_nodes[static_cast<size_t>(stack[--stack_size])];
            if (!intersect_aabb(node.bounds, ray, hit.t)) {
                continue;
            }
            if (node.count > 0) {
                for (int i = 0; i < node.count; ++i) {
                    const int index_offset = node.first + i;
                    if (index_offset < 0 || index_offset >= static_cast<int>(render_scene.mesh_instance_indices.size())) {
                        continue;
                    }
                    const int instance_index = render_scene.mesh_instance_indices[static_cast<size_t>(index_offset)];
                    if (instance_index < 0 || instance_index >= static_cast<int>(render_scene.mesh_instances.size())) {
                        continue;
                    }
                    const RenderScene::MeshInstance& instance = render_scene.mesh_instances[static_cast<size_t>(instance_index)];
                    if (instance.bvh_root >= 0 && intersect_aabb(instance.bounds, ray, hit.t)) {
                        found = intersect_blas(render_scene, ray, instance.bvh_root, hit) || found;
                    }
                }
            } else {
                if (node.left >= 0 && node.left < static_cast<int>(render_scene.tlas_nodes.size()) && stack_size < kTraversalStackSize) {
                    stack[stack_size++] = node.left;
                }
                if (node.right >= 0 && node.right < static_cast<int>(render_scene.tlas_nodes.size()) && stack_size < kTraversalStackSize) {
                    stack[stack_size++] = node.right;
                }
            }
        }
    }
    found = intersect_spheres(render_scene, ray, hit) || found;
    return found;
}

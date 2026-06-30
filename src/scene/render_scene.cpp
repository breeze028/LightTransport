#include "scene_internal.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lt {
namespace {

Aabb triangle_bounds(const Triangle& triangle) {
    constexpr float kBoundsEpsilon = 1.0e-4f;
    Aabb bounds;
    bounds.min = min(triangle.v0, min(triangle.v1, triangle.v2)) - Vec3{kBoundsEpsilon};
    bounds.max = max(triangle.v0, max(triangle.v1, triangle.v2)) + Vec3{kBoundsEpsilon};
    return bounds;
}

void expand(Aabb& bounds, Vec3 point) {
    bounds.min = min(bounds.min, point);
    bounds.max = max(bounds.max, point);
}

void expand(Aabb& bounds, const Aabb& other) {
    expand(bounds, other.min);
    expand(bounds, other.max);
}

float axis_value(Vec3 value, int axis) {
    return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
}

int longest_axis(Vec3 extent) {
    if (extent.y > extent.x && extent.y >= extent.z) {
        return 1;
    }
    if (extent.z > extent.x && extent.z >= extent.y) {
        return 2;
    }
    return 0;
}

int build_bvh_recursive(RenderScene& scene, std::vector<int>& indices, std::vector<BvhNode>& nodes, int first, int count) {
    BvhNode node;
    for (int i = first; i < first + count; ++i) {
        expand(node.bounds, triangle_bounds(scene.triangles[static_cast<size_t>(indices[static_cast<size_t>(i)])]));
    }

    const int node_index = static_cast<int>(nodes.size());
    nodes.push_back(node);
    if (count <= 4) {
        nodes[static_cast<size_t>(node_index)].first = first;
        nodes[static_cast<size_t>(node_index)].count = count;
        return node_index;
    }

    Aabb centroid_bounds;
    for (int i = first; i < first + count; ++i) {
        expand(centroid_bounds, scene.triangles[static_cast<size_t>(indices[static_cast<size_t>(i)])].centroid);
    }
    const int axis = longest_axis(centroid_bounds.max - centroid_bounds.min);
    const int mid = first + count / 2;
    std::nth_element(
        indices.begin() + first,
        indices.begin() + mid,
        indices.begin() + first + count,
        [&](int a, int b) {
            return axis_value(scene.triangles[static_cast<size_t>(a)].centroid, axis) <
                axis_value(scene.triangles[static_cast<size_t>(b)].centroid, axis);
        });

    nodes[static_cast<size_t>(node_index)].left = build_bvh_recursive(scene, indices, nodes, first, mid - first);
    nodes[static_cast<size_t>(node_index)].right = build_bvh_recursive(scene, indices, nodes, mid, first + count - mid);
    return node_index;
}

int build_tlas_recursive(RenderScene& scene, int first, int count) {
    BvhNode node;
    for (int i = first; i < first + count; ++i) {
        const int instance_index = scene.mesh_instance_indices[static_cast<size_t>(i)];
        if (instance_index >= 0 && instance_index < static_cast<int>(scene.mesh_instances.size())) {
            expand(node.bounds, scene.mesh_instances[static_cast<size_t>(instance_index)].bounds);
        }
    }

    const int node_index = static_cast<int>(scene.tlas_nodes.size());
    scene.tlas_nodes.push_back(node);
    if (count <= 2) {
        scene.tlas_nodes[static_cast<size_t>(node_index)].first = first;
        scene.tlas_nodes[static_cast<size_t>(node_index)].count = count;
        return node_index;
    }

    Aabb centroid_bounds;
    for (int i = first; i < first + count; ++i) {
        const int instance_index = scene.mesh_instance_indices[static_cast<size_t>(i)];
        const Aabb& bounds = scene.mesh_instances[static_cast<size_t>(instance_index)].bounds;
        expand(centroid_bounds, (bounds.min + bounds.max) * 0.5f);
    }
    const int axis = longest_axis(centroid_bounds.max - centroid_bounds.min);
    const int mid = first + count / 2;
    std::nth_element(
        scene.mesh_instance_indices.begin() + first,
        scene.mesh_instance_indices.begin() + mid,
        scene.mesh_instance_indices.begin() + first + count,
        [&](int a, int b) {
            const Aabb& bounds_a = scene.mesh_instances[static_cast<size_t>(a)].bounds;
            const Aabb& bounds_b = scene.mesh_instances[static_cast<size_t>(b)].bounds;
            return axis_value((bounds_a.min + bounds_a.max) * 0.5f, axis) <
                axis_value((bounds_b.min + bounds_b.max) * 0.5f, axis);
        });

    scene.tlas_nodes[static_cast<size_t>(node_index)].left = build_tlas_recursive(scene, first, mid - first);
    scene.tlas_nodes[static_cast<size_t>(node_index)].right = build_tlas_recursive(scene, mid, first + count - mid);
    return node_index;
}

} // namespace

RenderScene build_render_scene(const Scene& scene) {
    RenderScene render_scene;
    render_scene.spheres.reserve(scene.spheres.size());
    for (int sphere_index = 0; sphere_index < static_cast<int>(scene.spheres.size()); ++sphere_index) {
        const Sphere& sphere = scene.spheres[static_cast<size_t>(sphere_index)];
        if (sphere.material < 0 || sphere.material >= static_cast<int>(scene.materials.size()) || sphere.radius <= 0.0f) {
            continue;
        }
        render_scene.spheres.push_back({sphere.center, sphere.radius, sphere.material, sphere_index});
    }

    for (int mesh_index = 0; mesh_index < static_cast<int>(scene.meshes.size()); ++mesh_index) {
        const Mesh& mesh = scene.meshes[static_cast<size_t>(mesh_index)];
        const int triangle_index_begin = static_cast<int>(render_scene.triangles.size());
        const int index_begin = static_cast<int>(render_scene.triangle_indices.size());
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            const size_t i0 = static_cast<size_t>(mesh.indices[i]);
            const size_t i1 = static_cast<size_t>(mesh.indices[i + 1]);
            const size_t i2 = static_cast<size_t>(mesh.indices[i + 2]);
            if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() || i2 >= mesh.vertices.size()) {
                continue;
            }

            const Vec3 v0 = mesh.translation + scene_detail::rotate_xyz(mesh.vertices[i0] * mesh.scale, mesh.rotation);
            const Vec3 v1 = mesh.translation + scene_detail::rotate_xyz(mesh.vertices[i1] * mesh.scale, mesh.rotation);
            const Vec3 v2 = mesh.translation + scene_detail::rotate_xyz(mesh.vertices[i2] * mesh.scale, mesh.rotation);
            const Vec3 normal = normalize(cross(v1 - v0, v2 - v0));
            if (dot(normal, normal) <= 0.0f) {
                continue;
            }

            Vec3 n0 = normal;
            Vec3 n1 = normal;
            Vec3 n2 = normal;
            if (i0 < mesh.normals.size() && i1 < mesh.normals.size() && i2 < mesh.normals.size()) {
                n0 = normalize(scene_detail::rotate_xyz(mesh.normals[i0], mesh.rotation));
                n1 = normalize(scene_detail::rotate_xyz(mesh.normals[i1], mesh.rotation));
                n2 = normalize(scene_detail::rotate_xyz(mesh.normals[i2], mesh.rotation));
                if (dot(n0, n0) <= 0.0f || dot(n1, n1) <= 0.0f || dot(n2, n2) <= 0.0f) {
                    n0 = n1 = n2 = normal;
                }
            }

            const Vec2 uv0 = i0 < mesh.texcoords.size() ? mesh.texcoords[i0] : scene_detail::default_uv(mesh.vertices[i0]);
            const Vec2 uv1 = i1 < mesh.texcoords.size() ? mesh.texcoords[i1] : scene_detail::default_uv(mesh.vertices[i1]);
            const Vec2 uv2 = i2 < mesh.texcoords.size() ? mesh.texcoords[i2] : scene_detail::default_uv(mesh.vertices[i2]);
            const Vec3 edge1 = v1 - v0;
            const Vec3 edge2 = v2 - v0;
            const Vec2 duv1 = uv1 - uv0;
            const Vec2 duv2 = uv2 - uv0;
            Vec3 tangent;
            Vec3 bitangent;
            const float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
            if (std::fabs(determinant) > 1.0e-8f) {
                const float inverse_determinant = 1.0f / determinant;
                tangent = normalize((edge1 * duv2.y - edge2 * duv1.y) * inverse_determinant);
                bitangent = normalize((edge2 * duv1.x - edge1 * duv2.x) * inverse_determinant);
            }
            if (dot(tangent, tangent) <= 0.0f || dot(bitangent, bitangent) <= 0.0f) {
                const Vec3 up = std::fabs(normal.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
                tangent = normalize(cross(up, normal));
                bitangent = cross(normal, tangent);
            }

            const int triangle_index = static_cast<int>(render_scene.triangles.size());
            render_scene.triangles.push_back({
                v0, v1, v2, normal, n0, n1, n2, tangent, bitangent, (v0 + v1 + v2) / 3.0f,
                uv0, uv1, uv2, mesh.material, mesh_index,
            });
            render_scene.triangle_indices.push_back(triangle_index);

            Vec3 material_emission;
            bool material_emission_texture = false;
            if (mesh.material >= 0 && mesh.material < static_cast<int>(scene.materials.size()) && scene.materials[static_cast<size_t>(mesh.material)]) {
                const std::shared_ptr<Material>& material = scene.materials[static_cast<size_t>(mesh.material)];
                material_emission = material->emission;
                material_emission_texture = material->emission_texture != nullptr;
            }
            if ((mesh.light.enabled && mesh.light.intensity > 0.0f) ||
                scene_detail::has_light_emission(material_emission) ||
                material_emission_texture) {
                render_scene.light_triangle_indices.push_back(triangle_index);
            }
        }

        const int triangle_count = static_cast<int>(render_scene.triangles.size()) - triangle_index_begin;
        if (triangle_count > 0) {
            const int root = build_bvh_recursive(render_scene, render_scene.triangle_indices, render_scene.bvh_nodes, index_begin, triangle_count);
            RenderScene::MeshInstance instance;
            instance.bounds = render_scene.bvh_nodes[static_cast<size_t>(root)].bounds;
            instance.bvh_root = root;
            instance.mesh = mesh_index;
            render_scene.mesh_instance_indices.push_back(static_cast<int>(render_scene.mesh_instances.size()));
            render_scene.mesh_instances.push_back(instance);
        }
    }

    if (!render_scene.mesh_instance_indices.empty()) {
        build_tlas_recursive(render_scene, 0, static_cast<int>(render_scene.mesh_instance_indices.size()));
    }
    render_scene.flat_triangle_indices.resize(render_scene.triangles.size());
    std::iota(render_scene.flat_triangle_indices.begin(), render_scene.flat_triangle_indices.end(), 0);
    if (!render_scene.flat_triangle_indices.empty()) {
        build_bvh_recursive(render_scene, render_scene.flat_triangle_indices, render_scene.flat_bvh_nodes, 0, static_cast<int>(render_scene.flat_triangle_indices.size()));
    }
    return render_scene;
}

} // namespace lt

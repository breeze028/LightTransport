#include "lt/scene.h"

namespace lt {

Scene::Scene(const Scene& other)
    : camera(other.camera),
      environment(other.environment),
      uses_builtin_default_meshes(other.uses_builtin_default_meshes),
      meshes(other.meshes),
      spheres(other.spheres) {
    textures = other.textures;
    materials.reserve(other.materials.size());
    for (const std::shared_ptr<Material>& material : other.materials) {
        materials.push_back(material ? material->clone() : nullptr);
    }
}

Scene& Scene::operator=(const Scene& other) {
    if (this == &other) {
        return *this;
    }
    camera = other.camera;
    environment = other.environment;
    uses_builtin_default_meshes = other.uses_builtin_default_meshes;
    meshes = other.meshes;
    spheres = other.spheres;
    textures = other.textures;
    materials.clear();
    materials.reserve(other.materials.size());
    for (const std::shared_ptr<Material>& material : other.materials) {
        materials.push_back(material ? material->clone() : nullptr);
    }
    return *this;
}

} // namespace lt

#pragma once

#include "lt/scene.h"

namespace lt::scene_detail {

void add_triangle(Mesh& mesh, uint32_t a, uint32_t b, uint32_t c);
Vec2 default_uv(Vec3 vertex);
void ensure_default_uvs(Mesh& mesh);
Vec3 rotate_xyz(Vec3 point, Vec3 rotation);
void center_mesh_origin(Mesh& mesh);
bool has_light_emission(Vec3 emission);
LightComponent make_light_from_emission(Vec3 emission);

} // namespace lt::scene_detail

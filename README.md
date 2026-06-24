# LightTransport

An experimental light transport renderer playground. The initial framework includes:

- A mesh-only scene representation with cameras, materials, triangle meshes, and mesh light components.
- A CPU path tracer with BVH-accelerated triangle traversal for reference and algorithm prototyping.
- An optional CUDA path tracer backend over the same mesh scene representation, with GPU BVH traversal and a triangle light list for direct lighting.
- An ImGui + DirectX11 scene editor for loading, browsing, and interactively editing scenes.
- A CLI renderer that writes PPM images for regression checks and offline output.

## Build

Use one build directory per CMake generator. If a directory was already configured
with Visual Studio, do not reconfigure that same directory with Ninja; use a new
directory such as `build-ninja`, or delete the old build directory first.

Visual Studio 2022:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DLT_ENABLE_CUDA=ON
cmake --build build --config Release
```

Ninja:

```powershell
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DLT_ENABLE_CUDA=ON
cmake --build build-ninja
```

If CUDA is unavailable:

```powershell
cmake -S . -B build-ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DLT_ENABLE_CUDA=OFF
cmake --build build-ninja
```

To switch an existing build directory to a different generator, remove it first:

```powershell
Remove-Item -Recurse -Force build
```

## Run

```powershell
.\build\Release\lt_render.exe scenes\cornell.lt out.ppm --cuda
.\build\Release\lt_render.exe path\to\model.glb out.ppm --cpu
.\build\Release\lt_render.exe https://example.com/model.glb out.ppm --cpu
.\build\Release\lt_render.exe scenes\cornell.lt out.ppm --cuda --mis --mis-heuristic balance
.\build\Release\lt_render.exe scenes\cornell.lt xtoon.ppm --cpu --style x-toon --xtoon-mode highlight --style-samples 16 --style-depth 1
.\build\Release\lt_render.exe scenes\cornell.lt mixed.ppm --cpu --material-style blue x-toon --material-style white color-map --style-range 0 1
.\build\Release\lt_render.exe scenes\toon_material_test.lt hatch.ppm --cpu --material-style white cross-hatching --style-samples 64 --style-depth 1 --hatch-sets 3 --hatch-spacing 0.08 --hatch-width 0.012 --hatch-passthrough --hatch-shadow-only
.\build\Release\lt_editor.exe scenes\cornell.lt
```

For Ninja builds, the executables are under `build-ninja` instead:

```powershell
.\build-ninja\lt_render.exe scenes\cornell.lt out.ppm --cuda
.\build-ninja\lt_editor.exe scenes\cornell.lt
```

Editor controls:

- `W/A/S/D/Q/E`: move camera while the viewport is focused or hovered.
- Hold right mouse and drag in the viewport: look around.
- Left click in Select mode: select a mesh in the viewport.
- Move mode: drag the selected mesh in the camera plane.
- Rotate mode: drag the selected mesh to rotate it.
- Scale mode: drag the selected mesh to resize it.
- `O`: open a scene file.
- `Ctrl+R`: reset sample accumulation.
- `Space`: pause/resume preview accumulation.
- `Ctrl+S`: save the current scene.
- `Shift+A`: add a mesh.
- `Ctrl+D`: duplicate the selected mesh.
- `Delete`: delete the selected object.
- `Esc`: quit.

The editor layout follows Blender's broad shape: top menu and workspace strip, a left tool shelf, central rendered viewport with selection outlines and transform gizmos, right-side scene collection, lower-right properties tabs, and a bottom status bar.

Renderer backend, samples per frame, bounce depth, NPR style sampling, pause state, multiple importance sampling, and the MIS heuristic are controlled from Properties > Render. MIS is off by default; enable `--mis` on the command line or the checkbox in the editor.

The CLI renderer includes an experimental CPU implementation of direct-application stylized rendering from `papers/stylized-rendering`. NPR styles live on materials: `--style color-map`, `--style x-toon`, or `--style cross-hatching` applies a style to every material, while `--material-style material_name style` overrides individual materials. Color Map maps estimated exitant radiance brightness to a false-color gradient. X-Toon uses a procedural 2D toon ramp over diffuse tone plus a detail attribute selected with `--xtoon-mode constant|depth|silhouette|highlight`. Cross Hatching follows the paper supplemental setup: it selects view-tangent periodic plane sets from estimated radiance brightness and returns ink or paper color based on the current hit position. `--hatch-sets`, `--hatch-spacing`, `--hatch-width`, `--hatch-angle`, `--hatch-ink r g b`, `--hatch-paper r g b`, `--hatch-passthrough`, and `--hatch-shadow-only` control the hatch function. Shadow-only hatching tests visibility from the receiving surface to the first mesh light and lets non-shadowed or non-line regions pass through the original radiance, matching the cross-hatched shadow variant described for Fig. 1. `--style-samples N` controls the number of inner samples per style evaluation, while `--style-depth N` controls how many surface hits along a path are stylized. The editor exposes the per-material NPR controls in Properties > Material and the sampling budget in Properties > Render. Stylized rendering currently falls back to the CPU backend.

## Scene Format

The renderer can open native `.lt` scenes plus static glTF 2.0 `.glb`/`.gltf` scenes. On Windows, self-contained `.glb` scenes can also be opened from `http://` or `https://` URLs. glTF import supports triangle meshes, node transforms, vertex normals, UVs, base color textures, and metallic/roughness factors. Animation, skinning, morph targets, normal maps, and saving back to glTF are not implemented yet.

Scenes are plain text. Lines beginning with `#` are comments.

```text
camera px py pz  tx ty tz  fov_degrees
texture name  path.ppm
environment r g b  strength  [environment_texture_name]
material name  r g b  brdf  roughness metallic  [albedo_texture_name]
npr material_name  color_map value_min value_max
npr material_name  x_toon detail_mode steps shadow_r shadow_g shadow_b mid_r mid_g mid_b lit_r lit_g lit_b accent_r accent_g accent_b detail_strength detail_threshold detail_power depth_near depth_far
npr material_name  cross_hatching sets spacing width angle value_min value_max ink_r ink_g ink_b paper_r paper_g paper_b passthrough shadow_only
light mesh_name  color_r color_g color_b  intensity
sphere name material_name  cx cy cz  radius
mesh name material_name  tx ty tz  rx ry rz  sx sy sz  vertex_count triangle_count
vx vy vz
...
[normals vertex_count]
[nx ny nz]
...
i0 i1 i2
...
```

Native `.lt` texture lines support PPM `P3/P6` and Radiance `.hdr`; the editor and glTF importer can also decode PNG/JPEG textures on Windows. The optional `environment` line sets a constant environment or an equirectangular HDRI when a texture name is provided. Materials define surface color, an optional albedo texture, and BRDF. `brdf` is `lambertian`, `principled`, `mirror`, `dielectric`, or `conductor`; Principled mixes diffuse with GGX specular using roughness and metallic parameters, while dielectric uses the roughness slot as IOR in `.lt` files. Optional `npr` lines attach Color Map, X-Toon, or Cross Hatching settings to individual materials. `sphere` lines create analytic spheres with continuous normals. Mesh data can include an optional `normals vertex_count` block after vertices for smooth shading normals. A mesh becomes an area light when it has a `light` component; the emitted radiance is `color * intensity`. Older scene files that still put emission values on `material` lines are loaded by converting that legacy emission into light components on meshes using that material.

All renderable geometry is triangle mesh data. The built-in Cornell scene is also generated as meshes.

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
- `1`: CPU preview backend.
- `2`: CUDA preview backend when available.
- `O`: open a scene file.
- `Ctrl+R`: reset sample accumulation.
- `Space`: pause/resume preview accumulation.
- `Ctrl+S`: save the current scene.
- `Shift+A`: add a mesh.
- `Ctrl+D`: duplicate the selected mesh.
- `Delete`: delete the selected object.
- `Esc`: quit.

The editor layout follows Blender's broad shape: top menu and workspace strip, a left tool shelf, central rendered viewport with selection outlines and transform gizmos, right-side scene collection, lower-right properties tabs, and a bottom status bar.

## Scene Format

Scenes are plain text. Lines beginning with `#` are comments.

```text
camera px py pz  tx ty tz  fov_degrees
texture name  path.ppm
material name  r g b  brdf  roughness metallic  [albedo_texture_name]
light mesh_name  color_r color_g color_b  intensity
mesh name material_name  tx ty tz  rx ry rz  sx sy sz  vertex_count triangle_count
vx vy vz
...
i0 i1 i2
...
```

Textures currently load PPM `P3/P6` files. Materials define surface color, an optional albedo texture, and BRDF. `brdf` is `lambertian` or `principled`; Principled mixes diffuse with GGX specular using roughness and metallic parameters. A mesh becomes an area light when it has a `light` component; the emitted radiance is `color * intensity`. Older scene files that still put emission values on `material` lines are loaded by converting that legacy emission into light components on meshes using that material.

All renderable geometry is triangle mesh data. The built-in Cornell scene is also generated as meshes.

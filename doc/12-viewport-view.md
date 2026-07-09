# Viewport View

Viewport View 指编辑器中央视图的显示、交互和辅助渲染。它不是单一渲染器，而是两套输出路径叠在一起：

- Rendered：路径追踪器输出到 `Framebuffer::rgba`，再上传为 ImGui texture。
- Material Preview / Solid / Wireframe：D3D11 实时 raster preview，由 `SolidPreview` 管理。

此外 viewport 还负责 camera navigation、GPU picking、选中描边、gizmo、状态提示，以及给 SVGF 生成 rasterized G-buffer。

主要代码：

- `src/editor_win32.cpp`：UI 布局、viewport draw、输入、render task。
- `src/editor/solid_preview.h`
- `src/editor/solid_preview.cpp`

## 模式

枚举：

```text
ViewportPreviewMode::Rendered
ViewportPreviewMode::MaterialPreview
ViewportPreviewMode::Solid
ViewportPreviewMode::Wireframe
```

模式按钮在 `draw_viewport()` 附近绘制：

- Rendered：显示 path tracing 结果。
- Material Preview：D3D11 中读取材质颜色/纹理，近似 Blender 的材质预览。
- Solid：中性灰色 solid shading，适合看形体。
- Wireframe：只画线框，当前选中线框使用橙色风格。

只有 Rendered 模式会启动异步路径追踪任务。其它模式由 UI 线程同步调用 D3D11 preview。

## Rendered Preview

Rendered 流程：

```text
draw_viewport()
  -> launch_render_task()
       -> 复制 Scene / RenderSettings / Framebuffer
       -> 可选生成 rasterized SVGF G-buffer
       -> renderer->render()
  -> poll_render_result()
       -> 接收完成的 Framebuffer
       -> upload_preview_texture()
  -> ImGui::Image()
```

关键点：

- 渲染在后台任务中进行，避免 UI 阻塞。
- `reset_accumulation()` 会增加 render generation，旧任务结果不会覆盖新画面。
- `Framebuffer::rgba` 是最终显示用 LDR buffer。
- SVGF/TAA/PostAA 都在 renderer 写回 framebuffer 前完成。

Rendered 模式的 overlay（选中描边、viewport 按钮、提示文字）通过 ImGui draw list 叠加，不修改离线路径追踪输出。

## D3D11 实时预览

`SolidPreview` 持有 D3D11 资源：

```text
render target
depth target
MSAA resources
geometry buffers
material buffers/textures
object-id texture
outline texture
SVGF G-buffer targets
shaders / input layouts / states
```

主要入口：

```text
render_solid_preview()
render_svgf_gbuffer()
render_svgf_gbuffer_interop()
```

preview texture 尺寸跟 viewport image 尺寸一致。viewport resize 后需要重建 render target、depth、object-id 和 G-buffer。

## 几何与 Object ID

D3D11 preview 会把 Scene 转成 GPU geometry buffers。每个可选对象分配一个 `object_id`：

```text
0 = background / none
1..N = selectable object
```

同时维护映射：

```text
object_id -> SelectionKind + index
```

这个 ID 同时服务于：

- GPU picking。
- 选中描边。
- SVGF rasterized G-buffer 的 `object_id` AOV。

线框模式仍使用 solid fill 版本的 ID/depth pass 进行 picking，这样点击区域不会被线宽影响。

## GPU Picking

点击选择不再走 CPU 遍历三角形的慢路径，而是使用 GPU ID buffer：

```text
render object-id target + depth
  -> CopySubresourceRegion 鼠标附近小矩形到 staging texture
  -> Map staging texture
  -> 从中心向外找最近非 0 id
  -> id 映射回 Mesh/Sphere/Light
```

优点：

- 大 mesh 场景点击速度稳定。
- Rendered/Solid/Wireframe 使用同一套选择逻辑。
- 遮挡关系由 depth buffer 自然处理。

GPU picking 不可用时仍应保留 CPU fallback，保证编辑器可用。

## 选中描边

选中描边的目标是接近 Blender 的橙色 object outline，而不是只画 bounding box。

流程：

```text
render selected object id/depth
  -> fullscreen edge detect
  -> 输出透明橙色 outline texture
  -> ImGui overlay 合成
```

边缘检测主要看：

- selected id 与邻域 id 是否变化。
- depth 是否出现不连续。
- viewport 尺寸对应的 pixel step。

描边作为 overlay 绘制，不写回 `Framebuffer::rgba`。所以离线 render 输出不会包含选择框；这是编辑器 UI 状态。

## Solid、Material Preview 与 Wireframe

Solid：

- 使用 neutral color 和简单光照。
- 主要用于看 silhouette、法线和布局。

Material Preview：

- 上传材质基础参数和贴图。
- 只做实时近似，不追求路径追踪一致。
- 适合快速检查 texture binding、albedo、roughness 风格。

Wireframe：

- 使用 rasterizer wireframe state。
- 选中对象线框使用与 outline 同风格的橙色。
- 如果 `hide wireframe` 开启，选中对象线框也应隐藏，只保留其它 overlay。

这些模式都共享 camera transform、object ID、selection mapping 和 outline 资源。

## SVGF 的 Rasterized G-Buffer

SVGF 可以选择手动 path trace primary hit，也可以使用 rasterized G-buffer。编辑器里有开关：

```text
Rasterized G-Buffer
```

D3D11 G-buffer 输出：

```text
albedo
emission
normal
world_position + linear_depth
object_id
```

readback 路径：

```text
render_svgf_gbuffer()
  -> read_gbuffer_to_framebuffer()
  -> Framebuffer::AovBuffers
```

CUDA interop 路径：

```text
render_svgf_gbuffer_interop()
  -> D3D11 texture registered with CUDA
  -> map/unmap resource
  -> CUDA kernel reads or copies AOVs
```

关键要求：

- raster camera 和 path tracing primary ray 使用同一 camera/projection 约定。
- object_id、normal、depth 必须和当前 frame 对齐。
- resize 或 scene generation 变化时重建资源。
- interop resource 的 register/map/unmap/release 路径必须成对。

## Camera 与输入

Viewport 使用编辑器 camera：

- RMB look。
- WASD/QE fly。
- 鼠标滚轮/拖拽。
- gizmo drag。

相关函数：

```text
make_view_transform()
make_view_ray_from_screen()
project_point()
```

这些函数假定透视相机。修改相机模型时，不能只改 renderer 的 `make_camera_ray()`，还必须同步 viewport picking、gizmo 和 projection。

## Dirty 与 Generation

编辑器用 generation 避免旧任务覆盖新状态：

```text
render_generation
scene_generation
preview_generation
```

常见 dirty：

- Camera：重绘 viewport，Rendered 模式继续 progressive render。
- Geometry/Transform：更新 D3D11 geometry buffers、object ID、outline、render task。
- Material/Texture：更新 material preview buffers/textures。
- Environment：影响 Rendered 和 Material Preview 背景。

`reset_accumulation()` 是渲染刷新入口，不能只改字段不清空累积，否则会混入旧 frame。

## 验证

检查项：

- Rendered/Solid/Material Preview/Wireframe 切换后尺寸正确。
- 点击大 mesh、analytic sphere 和灯光都能选中。
- 选中 outline 不偏移、不闪烁、不只显示 bounding box。
- viewport resize 后 picking 和 outline 仍对齐。
- Wireframe 的 hide wireframe 能隐藏选中线框。
- Rasterized G-buffer 开关下 SVGF 不崩溃，AOV debug view 正常。

## 常见问题

- 点击选不中：object-id pass 没刷新、ID 映射丢失、staging readback 坐标和 viewport image 坐标不一致。
- 描边偏移：outline texture 尺寸与 viewport texture 不一致，或 fullscreen shader pixel step 错。
- Solid/Material Preview 黑屏：D3D11 render target/depth 没重建，或 shader compile 失败。
- Material Preview 贴图不对：纹理没有注册到 Scene textures，或 GPU material texture index 失效。
- Raster G-buffer 边缘锯齿明显：raster primary hit 与 path traced sample/jitter 不一致，需要在 SVGF/TAA 链路处理。

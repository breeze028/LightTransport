# CLI 与编辑器

## 相关专题

本页继续作为 CLI 参数和编辑器状态流的入口。涉及渲染辅助数据或 viewport 视觉的细节已经拆到专题文档：

- [10-irradiance-volume.md](10-irradiance-volume.md)：Render 面板中的 Irradiance Volume 控件、后台 bake、缓存和 dirty 规则。
- [11-lightmap.md](11-lightmap.md)：Lightmap 的编辑器按钮、overlay 预览、CPU/CUDA 上传路径。
- [12-viewport-view.md](12-viewport-view.md)：viewport 模式切换、GPU picking、选中描边、Material Preview 和 raster G-buffer。
- [13-svgf.md](13-svgf.md)：SVGF、rasterized G-buffer、AA mode、debug view 的 CLI/编辑器触达点。

## 命令行入口

`lt_render` 的位置参数：

```text
lt_render [scene_path] [output_path] [options...]
```

默认场景是 `scenes/cornell.lt`，默认输出是 `out.ppm`。输出路径支持 `.ppm` 和 `.png`；参数从 `argv[3]` 开始解析，因此只想传 option 时仍需写前两个位置参数。

### 当前选项

| 选项 | 作用 |
| --- | --- |
| `--cpu` / `--cuda` | 后端偏好 |
| `--mis` / `--no-mis` | MIS |
| `--mis-heuristic balance|power` | MIS 启发式 |
| `--accel flat|two-level` | 加速结构 |
| `--spp N` | 每帧每像素样本 |
| `--frames N` | 累积帧数 |
| `--size W H` | 输出尺寸 |
| `--style STYLE` | 为所有材质设置 NPR |
| `--material-style NAME STYLE` | 覆盖单个材质 |
| `--style-samples N` | NPR 内部样本 |
| `--style-depth N` | 风格化深度 |
| `--style-range MIN MAX` | Color Map/Hatching 亮度范围 |
| `--xtoon-mode MODE` | constant/depth/silhouette/highlight |
| `--hatch-sets N` | 线组数 |
| `--hatch-spacing F` | 线距 |
| `--hatch-width F` | 线宽 |
| `--hatch-angle F` | 基础角度，弧度 |
| `--hatch-ink R G B` | 墨色 |
| `--hatch-paper R G B` | 纸色 |
| `--hatch-passthrough` | 非线条区域保留原 radiance |
| `--hatch-shadow-only` | 仅阴影区域画线 |
| `--irradiance-volume` / `--no-irradiance-volume` | 启用/禁用辐照度体积间接光 |
| `--ivol-grid N` | 顶层八叉树分辨率（默认 7） |
| `--ivol-subgrid N` | 子网格分辨率（默认 3） |
| `--ivol-dir N` | 每探针方向数（默认 9） |
| `--ivol-bake-samples N` | 烘焙每方向样本数（默认 1） |
| `--ivol-bake-bounces N` | 烘焙间接光弹射次数（默认 4） |
| `--ivol-bounds-inset F` | 自动边界收缩量（默认 0.01） |
| `--ivol-principled-gi` / `--no-ivol-principled-gi` | 烘焙时使用 Principled 还是 Lambert GI |
| `--ivol-debug-probes` / `--no-ivol-debug-probes` | 可视化探针球 |
| `--ivol-probe-radius-scale F` | 调试探针半径缩放（默认 0.10） |
| `--ivol-cache PATH` | 设置缓存文件路径并启用缓存 |
| `--no-ivol-cache` | 禁用缓存读写 |
| `--ivol-auto-update` / `--no-ivol-auto-update` | 自动重新烘焙（编辑器用） |
| `--ivol-force-bake` | 忽略已有缓存，强制重新烘焙 |
| `--ivol-bounds min_x min_y min_z max_x max_y max_z` | 手动设置烘焙边界 |
| `--log-level trace|debug|info|warn|error|critical|off` | 同时设置控制台和文件日志等级 |
| `--log-file PATH` | 指定并启用日志文件 |
| `--no-log-file` | 关闭文件日志 |
| `--verbose` | 控制台和文件都输出 debug |
| `--quiet` | 关闭进度输出，并把控制台日志降到 warning |

解析器目前对未知参数静默忽略，对缺失值也不会统一报错。若 CLI 将用于自动化，建议后续增加错误收集、usage 和非零退出码。

### 日志输出

`lt_render` 默认把 `info` 及以上日志写到 stderr，把 `debug` 及以上日志写到 `logs/lt_render.log`。渲染进度仍写 stdout；脚本自动化时建议使用 `--quiet --log-file build/run.log`，这样终端输出较干净，但失败原因仍可追踪。

日志参数只影响观察方式，不改变加载 API 的错误返回语义。比如场景加载失败时，`SceneLoadResult::error` 仍然保留；CLI 同时会记录 fallback 到默认场景的 warning。

### 执行流程

`src/main.cpp`：

1. `parse_render_options()`。
2. 初始化日志。
3. `load_scene()`。
4. `apply_material_styles()`。
5. 设置辐照度体积缓存默认值（cache key 为场景路径，cache path 为 `<scene>.ivol`）。
6. 选择 CPU/CUDA；NPR 和辐照度体积烘焙强制 CPU。
7. 从 frame 0 渲染到 `--frames - 1`。
8. `lt::write_image()` 按输出扩展名写 `.ppm` 或 `.png`。

场景加载错误当前只打印警告并继续使用返回的 fallback Scene。

## 新增命令行参数

例如增加 `--max-radiance F`：

1. 在 `RenderSettings` 或 `RenderOptions` 加字段。
2. 在 `parse_render_options()` 增加分支，检查 `i + 参数数 < argc`。
3. 做范围夹紧，并明确非法值策略。
4. 若只是 CLI 层功能，在 `main.cpp` 应用；若渲染核心也需要，在 CPU/CUDA 中读取 setting。
5. 更新 README 和本文件。
6. 用缺失值、负值、合法值分别测试。

若多个选项共同配置一个功能，集中写辅助函数，参考 `apply_style()`，避免在 parser 中直接改 Scene。

## 离线输出

`lt::write_image()` 位于 `src/image_io.cpp`：

- `.ppm` 写 P3 文本 PPM。
- `.png` 通过 `stb_image_write` 写 8-bit RGBA PNG。
- 数据来自 `Framebuffer::rgba`，因此已经过固定 gamma 2.2 和 clamp。
- 不保留 HDR。
- 编辑器 File 菜单的 `Save Render Screenshot...` 复用同一套 PNG writer，保存当前 rendered framebuffer，不包含 ImGui overlay。

新增 EXR 或其他输出格式时继续扩展 `image_io` 模块：

- LDR 格式读取 `rgba` 或统一 tone-mapped buffer。
- EXR 读取 `accumulation / frame_count`。
- CUDA 当前不回传宿主侧 accumulation；EXR 支持还需增加设备 HDR 拷回，或明确只允许 CPU。
- 将输出格式按扩展名分派。
- 明确 RGBA 字节序；当前整数格式是 `0xAARRGGBB`，PNG writer 必须显式拆成 RGBA 字节，不能直接把整数内存传给 stb。

## 编辑器状态

`src/editor/editor_state.h::EditorState` 持有：

- Scene、Framebuffer、RenderSettings。
- CPU/CUDA 渲染器和当前 `IRenderer*`。
- dirty、frame index 和 render generation。
- 当前选择、工具模式、gizmo 拖动状态。
- 布局、viewport 和性能信息。

全局实例定义在 `editor_state.cpp`。编辑器当前是单窗口、全局状态架构，不是可复用的 UI 库。

## 异步场景加载与渲染

### 场景加载

`load_scene_file()` 用 `std::async` 调用 `lt::load_scene()`；`poll_scene_load_result()` 在 UI 循环中接收结果并调用 `set_scene()`。

同一时间只允许一个加载任务。

### 预览渲染

`launch_render_task()` 会复制：

- 当前 Scene。
- RenderSettings。
- Framebuffer。

后台调用渲染器，完成后 `poll_render_result()` 检查 generation。若用户在渲染期间修改场景，`reset_accumulation()` 会递增 generation，旧结果因此被丢弃。

共享风险：

- Scene copy 会 clone 材质但共享 Texture。
- CPU/CUDA renderer 对象本身由后台任务使用。当前只启动一个 render future，因此不会并发调用同一 renderer。
- 新增后台任务时不要并发修改 renderer 缓存或共享纹理。

## 编辑器主要扩展点

| 功能 | 函数/位置 |
| --- | --- |
| 重置与 dirty | `reset_accumulation()` |
| 后端选择 | `set_renderer()` |
| 场景替换 | `set_scene()` |
| 添加对象 | `add_cube_mesh()` 等 |
| 复制/删除 | `duplicate_selected()`、`delete_selected()` |
| 拾取 | `pick_object()` |
| gizmo | `handle_gizmo_drag()`、`draw_gizmo_overlay()` |
| NPR UI | `draw_npr_controls()` |
| 辐照度体积面板 | `draw_irradiance_volume_panel()`、`start_irradiance_volume_bake()` |
| 顶部菜单 | `draw_top_bar()` |
| 日志面板 | `draw_log_panel()`、`drain_editor_logs()`、`EditorState::logs` |
| Outliner | `draw_outliner()` |
| 属性面板 | `draw_properties()` |
| Viewport | `draw_viewport()` |
| 快捷键 | `handle_global_shortcuts()` |
| 总布局 | `draw_ui()` |

## 新增编辑器控件

例如给 Camera 增加 aperture：

```cpp
if (ImGui::DragFloat("Aperture", &g_editor.scene.camera.aperture,
                     0.01f, 0.0f, 32.0f)) {
    g_editor.scene.camera.aperture =
        std::clamp(g_editor.scene.camera.aperture, 0.0f, 32.0f);
    reset_accumulation(lt::RenderDirty::Camera);
}
```

规则：

- 像素结果变化必须 reset accumulation。
- 改 Camera 用 `Camera`。
- 改材质参数用 `Material`。
- 改纹理像素/资源用 `Texture`，通常也带 `Material` 或 `Environment`。
- 改顶点、索引、transform、sphere、mesh light 或影响灯列表的数据用 `Geometry`。
- 仅改 transform（位置、旋转、缩放）但不影响 BVH 结构时可用 `Transform`；否则用 `Geometry`。
- 改环境颜色/强度/方向用 `Environment`。
- 改辐照度体积参数或触发烘焙用 `IrradianceVolume`。
- 新增/删除 Mesh 后调用 `invalidate_mesh_bounds_cache()`，因为拾取与轮廓缓存依赖 Scene。
- 对 `.lt` 可保存对象修改时，将 `uses_builtin_default_meshes` 置 false，避免保存器省略内置 Mesh。
- 后台线程需要记录诊断信息时，只调用 `LT_LOG_*`；不要直接操作 ImGui。日志 observer 会先进入 pending queue，再由主线程绘制 Log 面板。

## 新增面板或对象类型

新增对象类型会影响的不只是 Outliner：

1. `SelectionKind` 和选中下标。
2. `has_selection()` 系列。
3. Outliner 绘制。
4. 属性面板。
5. Viewport picking。
6. 轮廓与 gizmo。
7. duplicate/delete。
8. `.lt` I/O。
9. `build_render_scene()`。
10. CPU/CUDA 求交和 GPU 打包。

如果新对象最终可以稳定三角化，优先在创建/导入时生成 Mesh，可避免扩展整条求交和编辑链。

## 后端选择与能力限制

编辑器 Render 面板的 `Renderer` 下拉框提供 `CPU Path Tracer`、`CUDA Megakernel Path Tracer`
和 `CUDA Wavefront Path Tracer` 三个互斥选项。CUDA 选项只有在以下条件满足时才可用：

- `cuda.available()`。
- 当前 Scene 未启用 NPR。
- 辐照度体积烘焙未在进行中（运行时查找 CUDA 支持）。

SVGF 是 denoiser 选择，不会反向禁用 CUDA Wavefront。CUDA Wavefront 使用内部 custom/wide BVH
layout，编辑器会固定 `Acceleration` 为 `Two-level BVH (Wavefront internal)`；切回 CPU 或 CUDA
Megakernel 后用户可重新选择 `Flat BVH` 或 `Two-level BVH`。

状态栏显示的是实际有效后端。新增 CPU-only 功能时应建立统一能力检查，而不是只在一个 UI 控件里禁用 CUDA，否则 CLI 和编辑器可能行为不一致。

### 编辑器辐照度体积面板

编辑器在 Render 面板中提供辐照度体积控件的完整集合，包括：

- 启用/禁用开关。
- 八叉树、方向和烘焙参数。
- 手动边界控件（`use_manual_bounds`）。
- 缓存路径和 key 输入框。
- 自动更新、强制烘焙和调试探针选项。
- "Bake Now" 按钮触发后台烘焙任务，进度显示在状态栏。

烘焙在 `EditorState` 管理的后台线程中运行。主循环通过 `poll_irradiance_volume_bake()` 检查进度并更新 UI。完成后自动触发累积重置和渲染刷新。

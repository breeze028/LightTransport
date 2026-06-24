# CLI 与编辑器

## 命令行入口

`lt_render` 的位置参数：

```text
lt_render [scene_path] [output_path] [options...]
```

默认场景是 `scenes/cornell.lt`，默认输出是 `out.ppm`。参数从 `argv[3]` 开始解析，因此只想传 option 时仍需写前两个位置参数。

### 当前选项

| 选项 | 作用 |
| --- | --- |
| `--cpu` / `--cuda` | 后端偏好 |
| `--mis` / `--no-mis` | MIS |
| `--mis-heuristic balance|power` | MIS 启发式 |
| `--accel auto|flat|two-level` | 加速结构 |
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

解析器目前对未知参数静默忽略，对缺失值也不会统一报错。若 CLI 将用于自动化，建议后续增加错误收集、usage 和非零退出码。

### 执行流程

`src/main.cpp`：

1. `parse_render_options()`。
2. `load_scene()`。
3. `apply_material_styles()`。
4. 选择 CPU/CUDA；NPR 强制 CPU。
5. 从 frame 0 渲染到 `--frames - 1`。
6. `write_ppm()` 写 ASCII P3。

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

`write_ppm()` 是 `src/main.cpp` 的匿名命名空间函数：

- 写 P3 文本 PPM。
- 数据来自 `Framebuffer::rgba`，因此已经过固定 gamma 2.2 和 clamp。
- 不保留 HDR。

新增 PNG/EXR 输出时建议建立独立 `image_io` 模块：

- LDR 格式读取 `rgba` 或统一 tone-mapped buffer。
- EXR 读取 `accumulation / frame_count`。
- CUDA 当前不回传宿主侧 accumulation；EXR 支持还需增加设备 HDR 拷回，或明确只允许 CPU。
- 将输出格式按扩展名分派。
- 明确 RGBA 字节序；当前整数格式是 `0xAARRGGBB`，Windows 小端内存供 D3D BGRA 纹理直接使用。

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
| 顶部菜单 | `draw_top_bar()` |
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
- 改环境颜色/强度/方向用 `Environment`。
- 新增/删除 Mesh 后调用 `invalidate_mesh_bounds_cache()`，因为拾取与轮廓缓存依赖 Scene。
- 对 `.lt` 可保存对象修改时，将 `uses_builtin_default_meshes` 置 false，避免保存器省略内置 Mesh。

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

`set_renderer(true)` 只有在以下条件满足时才会使用 CUDA：

- `cuda.available()`。
- 当前 Scene 未启用 NPR。

状态栏显示的是实际有效后端。新增 CPU-only 功能时应建立统一能力检查，而不是只在一个 UI 控件里禁用 CUDA，否则 CLI 和编辑器可能行为不一致。

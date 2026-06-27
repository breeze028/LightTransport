# 日志系统

日志系统位于公共 API [`include/lt/log.h`](../include/lt/log.h) 和实现文件 [`src/log.cpp`](../src/log.cpp)。它基于 spdlog 的同步 logger，第一版提供控制台、旋转文件和编辑器 observer 三类输出。

日志是观察能力，不替代现有错误返回语义。`SceneLoadResult::error`、纹理加载函数的 `std::string& error`、`save_scene()` 的 `error` 输出参数都继续保留；日志负责记录上下文、fallback 和可诊断细节。

## 公共 API

主要类型：

- `LogLevel`：`Trace`、`Debug`、`Info`、`Warn`、`Error`、`Critical`、`Off`。
- `LogConfig`：配置 logger 名称、控制台输出、文件输出、各 sink 等级、旋转文件大小和数量。
- `LogRecord`：传给 observer 的结构化记录，包含时间、等级、消息、源文件、行号、函数和线程 ID。

主要函数：

- `initialize_logging(config)`：应用层启动时调用。核心库本身不会主动创建日志文件。
- `shutdown_logging()` / `flush_logs()`：应用退出或测试结束时调用。
- `set_log_level(level)` / `log_level()`：运行时统一调整当前 logger 等级。
- `add_log_observer()` / `remove_log_observer()`：编辑器或测试可以订阅结构化日志。
- `parse_log_level()` / `log_level_name()`：CLI 和 UI 复用的字符串转换。

常用宏：

```cpp
LT_LOG_INFO("Loaded scene '{}' (meshes={})", path, scene.meshes.size());
LT_LOG_WARN("CUDA path tracer fallback to CPU: {}", reason);
LT_LOG_ERROR("Could not write {}", output_path);
```

宏会携带 `__FILE__`、`__LINE__` 和 `__func__`。格式化器支持按顺序替换 `{}`，并支持 `{{` / `}}` 输出字面量花括号。它是项目内轻量格式化器，不是完整 fmt 语法；不要使用 `{:03}`、`{name}` 这类高级格式。

## CLI 行为

`lt_render` 在 `main()` 一开始初始化日志：

- 默认控制台等级：`info`。
- 默认文件等级：`debug`。
- 默认文件路径：`logs/lt_render.log`。
- 默认旋转策略：5 MiB × 3。

命令行参数：

| 参数 | 行为 |
| --- | --- |
| `--log-level trace|debug|info|warn|error|critical|off` | 同时设置控制台和文件等级 |
| `--log-file PATH` | 指定日志文件路径并启用文件日志 |
| `--no-log-file` | 关闭文件日志 |
| `--verbose` | 控制台和文件都使用 `debug` |
| `--quiet` | 关闭渲染进度输出，并把控制台日志降到 `warn`；文件日志仍保留配置等级 |

渲染进度仍写 stdout；日志 sink 写 stderr 或文件。若要脚本里只检查结果文件，可以配合 `--quiet --log-file build/run.log`。

## 编辑器行为

编辑器启动时初始化：

- 文件日志：`logs/lt_editor.log`，等级 `debug`。
- observer：把所有通过当前 logger 等级的日志推入 `EditorState` 的线程安全 pending queue。
- UI：顶部菜单 `Window -> Log` 打开日志面板。

Log 面板支持：

- 最小等级过滤。
- Clear。
- Auto-scroll。
- hover 单条记录时显示源文件、行号和函数名。

异步加载线程只通过 observer 入队，不直接调用 ImGui。主线程每帧 `drain_editor_logs()`，再绘制 UI ring buffer。

## 应该在哪些位置加日志

优先记录边界事件：

- 应用启动、退出和主要配置。
- 场景、纹理、glTF、PBRT 加载开始、成功、失败和 fallback。
- 导入时跳过部分资源，例如 glTF 图片或 PBRT 纹理。
- CUDA fallback 到 CPU，尤其是显存分配、scene upload、kernel、copy-back 失败。
- 编辑器的 Open/Save/Load Texture/Load HDRI 等用户动作。

避免记录热路径：

- 不在 CPU 像素循环、sample 循环、BRDF 每次求值中打日志。
- 不在 CUDA device code 中使用日志宏。
- 不在高频 UI 控件拖动的每一帧刷 info；需要时用 debug，并只记录开始/结束或状态切换。

## 新增功能时的日志 checklist

- 新增加载器：入口 `info`，失败 `error`，可恢复资源缺失 `warn`。
- 新增后端或 GPU 上传路径：所有 fallback 必须 `warn` 一次，重复原因可降为 `debug`。
- 新增 CLI 参数：解析后在启动日志里能看出关键配置；无效值若继续 fallback，应 `warn`。
- 新增编辑器操作：用户触发的失败要同时保留 MessageBox/状态提示和日志记录。
- 新增长期后台任务：开始、完成、取消、失败都要有日志，避免只在 UI 状态里可见。

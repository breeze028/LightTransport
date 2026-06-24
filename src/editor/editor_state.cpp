#include "editor_state.h"

namespace lt::editor {

EditorState g_editor;
GpuPreview g_preview;
HWND g_hwnd = nullptr;
ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
IDXGISwapChain* g_swap_chain = nullptr;
ID3D11RenderTargetView* g_main_rtv = nullptr;
std::future<RenderResult> g_render_future;
SceneLoadTask g_load_task;
MeshBoundsCache g_bounds_cache;
PickSceneCache g_pick_cache;
UINT g_swap_chain_width = 0;
UINT g_swap_chain_height = 0;
bool g_window_minimized = false;
bool g_shutting_down = false;

} // namespace lt::editor

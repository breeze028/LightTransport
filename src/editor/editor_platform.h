#pragma once

#include <windows.h>

#include <string>

#include "lt/renderer.h"

namespace lt::editor {

enum class ViewportPreviewMode : int;

std::wstring widen(const std::string& text);
std::string narrow(const wchar_t* text);
void release_preview_texture();
void create_render_target();
void cleanup_render_target();
bool create_device(HWND hwnd);
void cleanup_device();
void release_editor_memory();
void reset_accumulation(lt::RenderDirty dirty = lt::RenderDirty::Render);
void release_rendered_preview_resources(bool release_renderer_cache, bool release_framebuffer);
void release_realtime_preview_resources(bool release_shaders);
void set_viewport_preview_mode(ViewportPreviewMode mode);

} // namespace lt::editor

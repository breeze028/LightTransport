#pragma once

#include <windows.h>

#include <string>

namespace lt::editor {

std::wstring widen(const std::string& text);
std::string narrow(const wchar_t* text);
void release_preview_texture();
void create_render_target();
void cleanup_render_target();
bool create_device(HWND hwnd);
void cleanup_device();
void release_editor_memory();

} // namespace lt::editor

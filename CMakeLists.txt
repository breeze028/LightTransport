cmake_minimum_required(VERSION 3.24)

project(LightTransport LANGUAGES CXX)

option(LT_ENABLE_CUDA "Build the CUDA path tracing backend" ON)
option(LT_BUILD_EDITOR "Build the realtime scene editor" ON)
option(LT_FETCH_IMGUI "Fetch Dear ImGui when building the editor" ON)
option(LT_FETCH_SPDLOG "Fetch spdlog when it is not provided by the system" ON)
option(LT_FETCH_STB "Fetch stb header-only image utilities when they are not provided by the system" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(spdlog CONFIG QUIET)
if(NOT spdlog_FOUND)
    if(LT_FETCH_SPDLOG)
        include(FetchContent)
        FetchContent_Declare(
            spdlog
            URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.zip
        )
        FetchContent_MakeAvailable(spdlog)
    else()
        message(FATAL_ERROR "spdlog was not found. Enable LT_FETCH_SPDLOG or provide spdlog::spdlog.")
    endif()
endif()

if(TARGET spdlog::spdlog_header_only)
    set(LT_SPDLOG_TARGET spdlog::spdlog_header_only)
else()
    set(LT_SPDLOG_TARGET spdlog::spdlog)
endif()

find_path(STB_INCLUDE_DIR stb_image_write.h)
if(NOT STB_INCLUDE_DIR)
    if(LT_FETCH_STB)
        include(FetchContent)
        FetchContent_Declare(
            stb
            URL https://github.com/nothings/stb/archive/refs/heads/master.zip
        )
        FetchContent_MakeAvailable(stb)
        set(STB_INCLUDE_DIR ${stb_SOURCE_DIR})
    else()
        message(FATAL_ERROR "stb_image_write.h was not found. Enable LT_FETCH_STB or provide STB_INCLUDE_DIR.")
    endif()
endif()

add_library(lt_core
    src/log.cpp
    src/cpu/path_tracer.cpp
    src/cpu/cuda_stub.cpp
    src/gltf_loader.cpp
    src/material.cpp
    src/pbrt/pbrt_loader.cpp
    src/scene.cpp
    src/scene/render_scene.cpp
    src/scene/scene_geometry.cpp
    src/scene/scene_io.cpp
    src/texture.cpp
)

target_include_directories(lt_core PUBLIC include)
target_include_directories(lt_core PRIVATE ${STB_INCLUDE_DIR})
target_link_libraries(lt_core PUBLIC ${LT_SPDLOG_TARGET})

if(WIN32)
    target_link_libraries(lt_core PUBLIC windowscodecs ole32 urlmon)
endif()

if(MSVC)
    target_compile_options(lt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
    target_compile_definitions(lt_core PUBLIC _SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)
else()
    target_compile_options(lt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

include(CheckLanguage)
if(LT_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        target_sources(lt_core PRIVATE src/gpu/cuda_path_tracer.cu)
        target_compile_definitions(lt_core PUBLIC LT_HAS_CUDA=1)
        set_source_files_properties(src/gpu/cuda_path_tracer.cu PROPERTIES LANGUAGE CUDA)
        set_property(TARGET lt_core PROPERTY CUDA_STANDARD 17)
        set_property(TARGET lt_core PROPERTY CUDA_ARCHITECTURES native)
        set_property(TARGET lt_core PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    else()
        message(WARNING "CUDA requested but no compiler was found. Building CPU-only.")
        target_compile_definitions(lt_core PUBLIC LT_HAS_CUDA=0)
    endif()
else()
    target_compile_definitions(lt_core PUBLIC LT_HAS_CUDA=0)
endif()

add_executable(lt_render
    src/main.cpp
    src/cli/render_options.cpp
)
target_link_libraries(lt_render PRIVATE lt_core)

if(LT_BUILD_EDITOR AND WIN32)
    include(FetchContent)
    if(LT_FETCH_IMGUI)
        FetchContent_Declare(
            imgui
            URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.8.zip
        )
        FetchContent_MakeAvailable(imgui)
    endif()

    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    )
    target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    target_compile_definitions(imgui PUBLIC IMGUI_DEFINE_MATH_OPERATORS)

    add_executable(lt_editor WIN32
        src/editor_win32.cpp
        src/editor/editor_platform.cpp
        src/editor/editor_state.cpp
    )
    target_link_libraries(lt_editor PRIVATE lt_core imgui user32 gdi32 comdlg32 shell32 d3d11 dxgi)
    target_compile_definitions(lt_editor PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

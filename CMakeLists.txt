cmake_minimum_required(VERSION 3.24)

project(LightTransport LANGUAGES CXX)

option(LT_ENABLE_CUDA "Build the CUDA path tracing backend" ON)
option(LT_BUILD_EDITOR "Build the realtime scene editor" ON)
option(LT_FETCH_IMGUI "Fetch Dear ImGui when building the editor" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

add_library(lt_core
    src/cpu_path_tracer.cpp
    src/material.cpp
    src/scene.cpp
    src/texture.cpp
)

target_include_directories(lt_core PUBLIC include)

if(MSVC)
    target_compile_options(lt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
else()
    target_compile_options(lt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

include(CheckLanguage)
if(LT_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        target_sources(lt_core PRIVATE src/cuda_path_tracer.cu)
        target_compile_definitions(lt_core PUBLIC LT_HAS_CUDA=1)
        set_source_files_properties(src/cuda_path_tracer.cu PROPERTIES LANGUAGE CUDA)
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

add_executable(lt_render src/main.cpp)
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

    add_executable(lt_editor WIN32 src/editor_win32.cpp)
    target_link_libraries(lt_editor PRIVATE lt_core imgui user32 gdi32 comdlg32 shell32 d3d11 dxgi)
    target_compile_definitions(lt_editor PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

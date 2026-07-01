cmake_minimum_required(VERSION 3.24)

project(LightTransport LANGUAGES CXX)

option(LT_ENABLE_CUDA "Build the CUDA path tracing backend" ON)
option(LT_BUILD_EDITOR "Build the realtime scene editor" ON)
option(LT_ENABLE_MATERIALX "Enable MaterialX/OpenPBR material interchange support when the SDK is available" ON)
option(LT_ENABLE_OPENIMAGEIO "Enable OpenImageIO texture decoding when the SDK is available" ON)
option(LT_ENABLE_OPENCOLORIO "Enable OpenColorIO color management when the SDK is available" ON)
option(LT_FETCH_IMGUI "Fetch Dear ImGui when building the editor" ON)
option(LT_FETCH_SPDLOG "Fetch spdlog when it is not provided by the system" ON)
option(LT_FETCH_STB "Fetch stb header-only image utilities when they are not provided by the system" ON)
option(LT_FETCH_XATLAS "Fetch xatlas when it is not provided by the system" ON)

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

if(LT_ENABLE_MATERIALX)
    find_package(MaterialX CONFIG QUIET)
endif()

if(LT_ENABLE_OPENIMAGEIO)
    find_package(OpenImageIO CONFIG QUIET)
endif()

if(LT_ENABLE_OPENCOLORIO)
    find_package(OpenColorIO CONFIG QUIET)
endif()

find_path(XATLAS_INCLUDE_DIR xatlas.h PATH_SUFFIXES xatlas)
find_library(XATLAS_LIBRARY NAMES xatlas)
set(LT_XATLAS_SOURCES)
if(NOT XATLAS_INCLUDE_DIR OR NOT XATLAS_LIBRARY)
    if(LT_FETCH_XATLAS)
        include(FetchContent)
        FetchContent_Declare(
            xatlas
            URL https://github.com/jpcy/xatlas/archive/refs/heads/master.zip
            URL_HASH SHA256=8D13E2D2C68B640417F182A7B3A8BBDF0E724A3B78A073E594F6D7941A73C971
        )
        FetchContent_MakeAvailable(xatlas)
        set(XATLAS_INCLUDE_DIR ${xatlas_SOURCE_DIR}/source/xatlas)
        set(LT_XATLAS_SOURCES ${xatlas_SOURCE_DIR}/source/xatlas/xatlas.cpp)
    else()
        message(FATAL_ERROR "xatlas was not found. Enable LT_FETCH_XATLAS or provide XATLAS_INCLUDE_DIR/XATLAS_LIBRARY.")
    endif()
endif()

add_library(lt_core
    ${LT_XATLAS_SOURCES}
    external/ufbx/ufbx.c
    src/log.cpp
    src/cpu/path_tracer.cpp
    src/cpu/cuda_stub.cpp
    src/fbx_loader.cpp
    src/gltf_loader.cpp
    src/material.cpp
    src/materialx_adapter.cpp
    src/pbrt/pbrt_loader.cpp
    src/scene.cpp
    src/scene/render_scene.cpp
    src/scene/scene_geometry.cpp
    src/scene/scene_io.cpp
    src/texture.cpp
)

target_include_directories(lt_core PUBLIC include)
target_include_directories(lt_core PRIVATE ${STB_INCLUDE_DIR})
target_include_directories(lt_core PRIVATE external/ufbx)
target_include_directories(lt_core PRIVATE ${XATLAS_INCLUDE_DIR})
target_link_libraries(lt_core PUBLIC ${LT_SPDLOG_TARGET})
if(XATLAS_LIBRARY)
    target_link_libraries(lt_core PUBLIC ${XATLAS_LIBRARY})
endif()
set_source_files_properties(external/ufbx/ufbx.c PROPERTIES LANGUAGE CXX)

if(MaterialX_FOUND)
    target_compile_definitions(lt_core PUBLIC LT_HAS_MATERIALX=1)
    foreach(materialx_target MaterialXCore MaterialXFormat MaterialXGenShader MaterialX::MaterialXCore MaterialX::MaterialXFormat)
        if(TARGET ${materialx_target})
            target_link_libraries(lt_core PUBLIC ${materialx_target})
        endif()
    endforeach()
else()
    target_compile_definitions(lt_core PUBLIC LT_HAS_MATERIALX=0)
    if(LT_ENABLE_MATERIALX)
        message(WARNING "MaterialX was not found. Building with the built-in OpenPBR-compatible material adapter only.")
    endif()
endif()

if(OpenImageIO_FOUND)
    target_compile_definitions(lt_core PUBLIC LT_HAS_OPENIMAGEIO=1)
    if(TARGET OpenImageIO::OpenImageIO)
        target_link_libraries(lt_core PUBLIC OpenImageIO::OpenImageIO)
    endif()
else()
    target_compile_definitions(lt_core PUBLIC LT_HAS_OPENIMAGEIO=0)
    if(LT_ENABLE_OPENIMAGEIO)
        message(WARNING "OpenImageIO was not found. Building with the built-in WIC/HDR/EXR texture loader fallback.")
    endif()
endif()

if(OpenColorIO_FOUND)
    target_compile_definitions(lt_core PUBLIC LT_HAS_OPENCOLORIO=1)
    if(TARGET OpenColorIO::OpenColorIO)
        target_link_libraries(lt_core PUBLIC OpenColorIO::OpenColorIO)
    endif()
else()
    target_compile_definitions(lt_core PUBLIC LT_HAS_OPENCOLORIO=0)
    if(LT_ENABLE_OPENCOLORIO)
        message(WARNING "OpenColorIO was not found. Building with explicit sRGB/raw texture role conversion fallback.")
    endif()
endif()

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
    target_link_libraries(lt_editor PRIVATE lt_core imgui user32 gdi32 comdlg32 shell32 d3d11 dxgi psapi)
    target_compile_definitions(lt_editor PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

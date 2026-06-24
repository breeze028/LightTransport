#include "cli/render_options.h"

#include <fstream>
#include <iostream>

namespace {

bool write_ppm(const std::string& path, const lt::Framebuffer& framebuffer) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << "P3\n" << framebuffer.width << ' ' << framebuffer.height << "\n255\n";
    for (uint32_t color : framebuffer.rgba) {
        output << ((color >> 16u) & 0xffu) << ' '
            << ((color >> 8u) & 0xffu) << ' '
            << (color & 0xffu) << '\n';
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    lt::cli::RenderOptions options = lt::cli::parse_render_options(argc, argv);
    lt::SceneLoadResult loaded = lt::load_scene(options.scene_path);
    if (!loaded.error.empty()) {
        std::cerr << loaded.error << "\nUsing fallback default scene.\n";
    }
    lt::cli::apply_material_styles(options, loaded.scene, std::cerr);

    lt::CpuPathTracer cpu;
    lt::CudaPathTracer cuda;
    lt::IRenderer* renderer = &cpu;
    if (options.prefer_cuda && lt::stylized_rendering_enabled(options.settings, loaded.scene)) {
        std::cout << "Stylized rendering is currently implemented on CPU; using CPU renderer.\n";
    } else if (options.prefer_cuda && cuda.available()) {
        renderer = &cuda;
    }

    lt::Framebuffer framebuffer;
    framebuffer.resize(options.settings.width, options.settings.height);
    const uint32_t frames = options.settings.frame_index + 1u;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        options.settings.frame_index = frame;
        options.settings.dirty = frame == 0 ? lt::RenderDirty::All : lt::RenderDirty::None;
        renderer->render(loaded.scene, options.settings, framebuffer);
        std::cout << "\r" << renderer->name() << " frame " << (frame + 1u) << "/" << frames << std::flush;
    }
    std::cout << "\n";

    if (!write_ppm(options.output_path, framebuffer)) {
        std::cerr << "Could not write " << options.output_path << "\n";
        return 1;
    }
    std::cout << "Wrote " << options.output_path << "\n";
    return 0;
}

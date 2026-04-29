#include "lt/renderer.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

namespace {

bool write_ppm(const std::string& path, const lt::Framebuffer& framebuffer) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "P3\n" << framebuffer.width << ' ' << framebuffer.height << "\n255\n";
    for (uint32_t c : framebuffer.rgba) {
        out << ((c >> 16u) & 0xffu) << ' '
            << ((c >> 8u) & 0xffu) << ' '
            << (c & 0xffu) << '\n';
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const std::string scene_path = argc > 1 ? argv[1] : "scenes/cornell.lt";
    const std::string output_path = argc > 2 ? argv[2] : "out.ppm";

    lt::RenderSettings settings;
    settings.width = 1280;
    settings.height = 720;
    settings.samples_per_pixel = 1;
    settings.max_bounces = 6;

    bool prefer_cuda = false;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cuda") {
            prefer_cuda = true;
        } else if (arg == "--cpu") {
            prefer_cuda = false;
        } else if (arg == "--mis") {
            settings.use_mis = true;
        } else if (arg == "--no-mis") {
            settings.use_mis = false;
        } else if (arg == "--primary-hit-cache") {
            settings.use_primary_hit_cache = true;
        } else if (arg == "--no-primary-hit-cache") {
            settings.use_primary_hit_cache = false;
        } else if (arg == "--mis-heuristic" && i + 1 < argc) {
            const std::string heuristic = argv[++i];
            settings.mis_heuristic = heuristic == "balance" ? lt::MisHeuristic::Balance : lt::MisHeuristic::Power;
        } else if (arg == "--accel" && i + 1 < argc) {
            const std::string accel = argv[++i];
            if (accel == "two-level" || accel == "twolevel" || accel == "tlas") {
                settings.acceleration_structure = lt::AccelerationStructure::TwoLevel;
            } else if (accel == "flat") {
                settings.acceleration_structure = lt::AccelerationStructure::Flat;
            } else {
                settings.acceleration_structure = lt::AccelerationStructure::Auto;
            }
        } else if (arg == "--spp" && i + 1 < argc) {
            settings.samples_per_pixel = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--frames" && i + 1 < argc) {
            settings.frame_index = static_cast<uint32_t>(std::max(0, std::atoi(argv[++i]) - 1));
        } else if (arg == "--size" && i + 2 < argc) {
            settings.width = std::max(1, std::atoi(argv[++i]));
            settings.height = std::max(1, std::atoi(argv[++i]));
        }
    }

    lt::SceneLoadResult loaded = lt::load_scene(scene_path);
    if (!loaded.error.empty()) {
        std::cerr << loaded.error << "\nUsing fallback default scene.\n";
    }

    lt::CpuPathTracer cpu;
    lt::CudaPathTracer cuda;
    lt::IRenderer* renderer = &cpu;
    if (prefer_cuda && cuda.available()) {
        renderer = &cuda;
    }

    lt::Framebuffer framebuffer;
    framebuffer.resize(settings.width, settings.height);
    const uint32_t frames = settings.frame_index + 1u;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        settings.frame_index = frame;
        settings.dirty = frame == 0 ? lt::RenderDirty::All : lt::RenderDirty::None;
        renderer->render(loaded.scene, settings, framebuffer);
        std::cout << "\r" << renderer->name() << " frame " << (frame + 1u) << "/" << frames << std::flush;
    }
    std::cout << "\n";

    if (!write_ppm(output_path, framebuffer)) {
        std::cerr << "Could not write " << output_path << "\n";
        return 1;
    }
    std::cout << "Wrote " << output_path << "\n";
    return 0;
}

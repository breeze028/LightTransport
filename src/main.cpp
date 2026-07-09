#include "cli/render_options.h"

#include <cstdio>
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

struct LoggingScope {
    ~LoggingScope() {
        lt::flush_logs();
        lt::shutdown_logging();
    }
};

template <size_t N>
void copy_setting_text(char (&target)[N], const std::string& value) {
    std::snprintf(target, N, "%s", value.c_str());
}

void set_irradiance_volume_cache_defaults(lt::RenderSettings& settings, const std::string& scene_path) {
    if (settings.irradiance_volume_cache_key[0] == '\0') {
        copy_setting_text(settings.irradiance_volume_cache_key, scene_path);
    }
    if (settings.irradiance_volume_cache_path[0] == '\0' && !scene_path.empty()) {
        copy_setting_text(settings.irradiance_volume_cache_path, scene_path + ".ivol");
    }
}

void set_lightmap_cache_defaults(lt::RenderSettings& settings, const std::string& scene_path) {
    if (settings.lightmap_cache_key[0] == '\0') {
        copy_setting_text(settings.lightmap_cache_key, scene_path);
    }
    if (settings.lightmap_cache_path[0] == '\0' && !scene_path.empty()) {
        copy_setting_text(settings.lightmap_cache_path, scene_path + ".lmap");
    }
}

} // namespace

int main(int argc, char** argv) {
    lt::cli::RenderOptions options = lt::cli::parse_render_options(argc, argv);
    lt::LogConfig log_config;
    log_config.logger_name = "lt_render";
    log_config.enable_console = options.log_console_level != lt::LogLevel::Off;
    log_config.console_level = options.log_console_level;
    log_config.enable_file = options.log_file_enabled && !options.log_file_path.empty();
    log_config.file_level = options.log_file_level;
    log_config.file_path = options.log_file_path;
    lt::initialize_logging(log_config);
    LoggingScope logging_scope;

    LT_LOG_INFO("Starting render: scene='{}' output='{}'", options.scene_path, options.output_path);
    lt::SceneLoadResult loaded = lt::load_scene(options.scene_path);
    if (!loaded.error.empty()) {
        LT_LOG_WARN("{}; using fallback default scene", loaded.error);
    }
    lt::cli::apply_material_styles(options, loaded.scene, std::cerr);
    set_irradiance_volume_cache_defaults(options.settings, options.scene_path);
    set_lightmap_cache_defaults(options.settings, options.scene_path);

    lt::CpuPathTracer cpu;
    lt::CudaPathTracer cuda;
    lt::IRenderer* renderer = &cpu;
    if (options.prefer_cuda && lt::stylized_rendering_enabled(options.settings, loaded.scene)) {
        LT_LOG_WARN("Stylized rendering is currently implemented on CPU; using CPU renderer");
    } else if (options.prefer_cuda && cuda.available()) {
        renderer = &cuda;
    }
    LT_LOG_INFO("Using renderer: {}", renderer->name());

    lt::Framebuffer framebuffer;
    framebuffer.resize(options.settings.width, options.settings.height);
    const uint32_t frames = options.settings.frame_index + 1u;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        options.settings.frame_index = frame;
        options.settings.dirty = frame == 0 ? lt::RenderDirty::All : lt::RenderDirty::None;
        if (lt::temporal_jitter_enabled(options.settings)) {
            const lt::Vec2 jitter = lt::temporal_jitter(frame);
            options.settings.camera_jitter_x = jitter.x;
            options.settings.camera_jitter_y = jitter.y;
        } else {
            options.settings.camera_jitter_x = 0.0f;
            options.settings.camera_jitter_y = 0.0f;
        }
        renderer->render(loaded.scene, options.settings, framebuffer);
        if (!options.quiet) {
            std::cout << "\r" << renderer->name() << " frame " << (frame + 1u) << "/" << frames << std::flush;
        }
    }
    if (!options.quiet) {
        std::cout << "\n" << std::flush;
    }

    if (!write_ppm(options.output_path, framebuffer)) {
        LT_LOG_ERROR("Could not write {}", options.output_path);
        return 1;
    }
    LT_LOG_INFO("Wrote {}", options.output_path);
    if (!options.quiet) {
        std::cout << "Wrote " << options.output_path << "\n";
    }
    return 0;
}

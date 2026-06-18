Ray make_camera_ray(const Camera& camera, int x, int y, const RenderSettings& settings, Rng& rng) {
    const float aspect = static_cast<float>(settings.width) / static_cast<float>(settings.height);
    const float half_height = std::tan(camera.fov_degrees * kPi / 360.0f);
    const float half_width = aspect * half_height;
    const Vec3 forward = normalize(camera.target - camera.position);
    const float right_sign = camera.right_sign < 0.0f ? -1.0f : 1.0f;
    const Vec3 right = normalize(cross(forward, camera.up)) * right_sign;
    const Vec3 up = cross(right, forward) * right_sign;
    const float u = ((static_cast<float>(x) + rng.next_float()) / static_cast<float>(settings.width) * 2.0f - 1.0f) * half_width;
    const float v = (1.0f - (static_cast<float>(y) + rng.next_float()) / static_cast<float>(settings.height) * 2.0f) * half_height;
    return {camera.position, normalize(forward + right * u + up * v)};
}


#pragma once

#include "pnp_solver.hpp"

#include <opencv2/core.hpp>


// yaw/pitch 角度单位为度，距离及平移单位为米
struct PoseGroundTruth
{
    bool has_distance = false;
    double distance_m = 0.0;
    bool has_yaw = false;
    double yaw_deg = 0.0;
    bool has_pitch = false;
    double pitch_deg = 0.0;
    bool has_tvec = false;
    cv::Vec3d tvec{0.0, 0.0, 0.0};
    bool has_rvec = false;
    cv::Vec3d rvec{0.0, 0.0, 0.0};
};

struct PoseError
{
    bool has_distance_error = false;
    double distance_error_m = 0.0;
    bool has_yaw_error = false;
    double yaw_error_deg = 0.0;
    bool has_pitch_error = false;
    double pitch_error_deg = 0.0;
    bool has_tvec_error = false;
    cv::Vec3d tvec_error{0.0, 0.0, 0.0};
    bool has_rotation_error = false;
    double rotation_error_deg = 0.0;
};

// 对已有真值字段计算误差；缺失字段保持对应 has_*_error=false。
PoseError compareToGroundTruth(const PnPResult &estimate, const PoseGroundTruth &truth);

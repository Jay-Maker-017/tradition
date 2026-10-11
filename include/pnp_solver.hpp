#pragma once

#include "armor.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

// 相机标定参数
// 长度单位为米，像素坐标以标定图像左上角为原点
struct CameraParams
{
    cv::Mat camera_matrix; // CV_64F, 3x3
    cv::Mat dist_coeffs;   // CV_64F, 1xN
    cv::Size image_size;
    double calibration_reprojection_error = 0.0; // 标定阶段 RMS，仅供参考
};

// 读取 JSON 标定配置
// 读取失败时返回工程内置的相机参数并打印警告
CameraParams loadCameraParams(const std::string &path);

// 若实际图片尺寸与标定尺寸不同，按 x/y 比例缩放焦距及主点并打印警告
CameraParams adaptCameraParams(const CameraParams &calibrated, const cv::Size &image_size);

struct PnPResult
{
    bool valid = false;
    std::string message;
    int armor_index = -1;
    int size = -1;
    cv::Vec3d rvec{0.0, 0.0, 0.0};
    cv::Vec3d tvec{0.0, 0.0, 0.0};
    double distance_m = 0.0;
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
    double reprojection_error_px = 0.0;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point2f> reprojected_points;
};

class PnPSolver
{
public:
    explicit PnPSolver(CameraParams camera_params);

    PnPResult solve(const Armor &armor, int armor_index = -1) const;
    std::vector<PnPResult> solveAll(const std::vector<Armor> &armors) const;
    const CameraParams &cameraParams() const { return camera_params_; }

    static constexpr double kSmallArmorWidthM = 0.135;
    static constexpr double kSmallArmorHeightM = 0.055;
    static constexpr double kLargeArmorWidthM = 0.225;
    static constexpr double kLargeArmorHeightM = 0.055;
    static constexpr double kReprojectionWarningThresholdPx = 2.0;

private:
    CameraParams camera_params_;
    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
};

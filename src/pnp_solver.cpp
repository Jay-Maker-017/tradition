#include "pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace
{
CameraParams makeDefaultCameraParams()
{
    CameraParams params;
    params.camera_matrix = (cv::Mat_<double>(3, 3) <<
        2362.2795952460974, 0.0, 679.17513921636282,
        0.0, 2336.1916035376717, 529.06858905837635,
        0.0, 0.0, 1.0);
    params.dist_coeffs = (cv::Mat_<double>(1, 5) <<
        -0.098561592108101559, 0.11173066531932681,
        -0.0038352943354436304, -0.0020421443354436304, 0.0);
    params.image_size = cv::Size(1440, 1080);
    params.calibration_reprojection_error = 0.1448;
    return params;
}

std::vector<double> readNumberSequence(const cv::FileNode &node)
{
    if (node.type() != cv::FileNode::SEQ)
        throw std::runtime_error("期望一个数值数组");
    std::vector<double> values;
    for (const cv::FileNode &item : node)
    {
        if (item.type() != cv::FileNode::INT && item.type() != cv::FileNode::REAL)
            throw std::runtime_error("数组包含非数值元素");
        values.push_back(static_cast<double>(item));
    }
    return values;
}

std::filesystem::path findConfigPath(const std::string &path)
{
    const std::filesystem::path requested(path);
    if (std::filesystem::exists(requested))
        return requested;
    if (requested.is_relative())
    {
        const std::filesystem::path from_build = std::filesystem::path("..") / requested;
        if (std::filesystem::exists(from_build))
            return from_build;
    }
    return requested;
}

cv::Vec3d matToVec3d(const cv::Mat &input)
{
    cv::Mat converted;
    input.reshape(1, 3).convertTo(converted, CV_64F);
    return {converted.at<double>(0, 0), converted.at<double>(1, 0), converted.at<double>(2, 0)};
}

std::vector<cv::Point3f> makePlanarArmorPoints(double width_m, double height_m)
{
    const float half_width = static_cast<float>(width_m / 2.0);
    const float half_height = static_cast<float>(height_m / 2.0);
    // 对象点严格匹配 armorCorners() 的 LT, RT, RB, LB 顺序。
    // 坐标系：x 向右、y 向下、z 向前（相机坐标系约定），原点为板中心。
    return {{-half_width, -half_height, 0.0f},
            { half_width, -half_height, 0.0f},
            { half_width,  half_height, 0.0f},
            {-half_width,  half_height, 0.0f}};
}
} // namespace

CameraParams loadCameraParams(const std::string &path)
{
    const CameraParams defaults = makeDefaultCameraParams();
    try
    {
        const std::filesystem::path config_path = findConfigPath(path);
        cv::FileStorage storage(config_path.string(), cv::FileStorage::READ);
        if (!storage.isOpened())
            throw std::runtime_error("无法打开配置文件: " + config_path.string());

        cv::FileNode camera = storage["camera"];
        if (camera.empty())
            camera = storage.root();

        CameraParams params;
        int image_width = 0;
        int image_height = 0;
        camera["image_width"] >> image_width;
        camera["image_height"] >> image_height;
        camera["calib_reprojection_error"] >> params.calibration_reprojection_error;
        const std::vector<double> matrix_values = readNumberSequence(camera["camera_matrix"]);
        const std::vector<double> distortion_values = readNumberSequence(camera["distort_coeffs"]);
        if (matrix_values.size() != 9 || distortion_values.empty())
            throw std::runtime_error("camera_matrix 需有 9 项且 distort_coeffs 不可为空");
        if (image_width <= 0 || image_height <= 0)
            throw std::runtime_error("标定图像分辨率无效");

        params.camera_matrix = cv::Mat(3, 3, CV_64F);
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                params.camera_matrix.at<double>(row, col) = matrix_values[static_cast<std::size_t>(row * 3 + col)];
        params.dist_coeffs = cv::Mat(1, static_cast<int>(distortion_values.size()), CV_64F);
        for (std::size_t i = 0; i < distortion_values.size(); ++i)
            params.dist_coeffs.at<double>(0, static_cast<int>(i)) = distortion_values[i];
        params.image_size = cv::Size(image_width, image_height);

        if (!std::isfinite(params.calibration_reprojection_error) ||
            params.camera_matrix.at<double>(0, 0) <= 0.0 ||
            params.camera_matrix.at<double>(1, 1) <= 0.0)
            throw std::runtime_error("相机参数数值无效");
        return params;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[PnP 警告] " << error.what()
                  << "；使用内置相机参数。" << std::endl;
        return defaults;
    }
}

CameraParams adaptCameraParams(const CameraParams &calibrated, const cv::Size &image_size)
{
    CameraParams adapted = calibrated;
    adapted.camera_matrix = calibrated.camera_matrix.clone();
    adapted.dist_coeffs = calibrated.dist_coeffs.clone();
    if (image_size.width <= 0 || image_size.height <= 0 || calibrated.image_size.width <= 0 ||
        calibrated.image_size.height <= 0)
    {
        std::cerr << "[PnP 警告] 图片或标定分辨率无效，无法缩放相机内参。" << std::endl;
        return adapted;
    }
    if (image_size == calibrated.image_size)
        return adapted;

    const double scale_x = static_cast<double>(image_size.width) / calibrated.image_size.width;
    const double scale_y = static_cast<double>(image_size.height) / calibrated.image_size.height;
    adapted.camera_matrix.at<double>(0, 0) *= scale_x;
    adapted.camera_matrix.at<double>(0, 2) *= scale_x;
    adapted.camera_matrix.at<double>(1, 1) *= scale_y;
    adapted.camera_matrix.at<double>(1, 2) *= scale_y;
    adapted.image_size = image_size;
    std::cerr << "[PnP 警告] 图片分辨率 " << image_size.width << "x" << image_size.height
              << " 与标定分辨率 " << calibrated.image_size.width << "x"
              << calibrated.image_size.height << " 不同，已按比例缩放相机内参。" << std::endl;
    return adapted;
}

PnPSolver::PnPSolver(CameraParams camera_params)
    : camera_params_(std::move(camera_params)),
      small_armor_points_(makePlanarArmorPoints(kSmallArmorWidthM, kSmallArmorHeightM)),
      large_armor_points_(makePlanarArmorPoints(kLargeArmorWidthM, kLargeArmorHeightM))
{
    if (camera_params_.camera_matrix.rows != 3 || camera_params_.camera_matrix.cols != 3 ||
        camera_params_.camera_matrix.type() != CV_64F || camera_params_.dist_coeffs.empty())
        throw std::invalid_argument("PnPSolver 收到的相机参数格式无效");
}

PnPResult PnPSolver::solve(const Armor &armor, int armor_index) const
{
    PnPResult result;
    result.armor_index = armor_index;
    result.size = armor.size;
    if (!armor.has_type)
    {
        result.message = "网络未确认装甲板类型";
        return result;
    }

    const std::vector<cv::Point3f> *object_points = nullptr;
    if (armor.size == 0)
        object_points = &small_armor_points_;
    else if (armor.size == 1)
        object_points = &large_armor_points_;
    else
    {
        result.message = "装甲板大小未知或无效";
        return result;
    }

    result.image_points = armorCorners(armor.left, armor.right);
    if (result.image_points.size() != 4)
    {
        result.message = "图像角点数量不是 4";
        return result;
    }

    try
    {
        cv::Mat rvec;
        cv::Mat tvec;
        const bool solved = cv::solvePnP(*object_points, result.image_points,
                                         camera_params_.camera_matrix,
                                         camera_params_.dist_coeffs, rvec, tvec,
                                         false, cv::SOLVEPNP_IPPE);
        if (!solved)
        {
            result.message = "cv::solvePnP 未找到有效解";
            return result;
        }

        result.rvec = matToVec3d(rvec);
        result.tvec = matToVec3d(tvec);
        result.distance_m = cv::norm(result.tvec);
        result.yaw_deg = std::atan2(result.tvec[0], result.tvec[2]) * 180.0 / CV_PI;
        result.pitch_deg = std::atan2(result.tvec[1], result.tvec[2]) * 180.0 / CV_PI;

        std::vector<cv::Point2f> projected;
        cv::projectPoints(*object_points, rvec, tvec, camera_params_.camera_matrix,
                          camera_params_.dist_coeffs, projected);
        result.reprojected_points = projected;
        double squared_error_sum = 0.0;
        for (std::size_t i = 0; i < projected.size(); ++i)
        {
            const cv::Point2f difference = projected[i] - result.image_points[i];
            squared_error_sum += difference.dot(difference);
        }
        result.reprojection_error_px = std::sqrt(squared_error_sum / projected.size());
        result.valid = true;
        if (result.reprojection_error_px > kReprojectionWarningThresholdPx)
        {
            std::cerr << "[PnP 警告] Armor " << (armor_index + 1)
                      << " 重投影 RMS=" << result.reprojection_error_px
                      << " px，超过 " << kReprojectionWarningThresholdPx
                      << " px；保留该解算结果。" << std::endl;
        }
    }
    catch (const cv::Exception &error)
    {
        result.message = std::string("OpenCV PnP 异常: ") + error.what();
    }
    return result;
}

std::vector<PnPResult> PnPSolver::solveAll(const std::vector<Armor> &armors) const
{
    std::vector<PnPResult> results;
    results.reserve(armors.size());
    for (std::size_t i = 0; i < armors.size(); ++i)
        results.push_back(solve(armors[i], static_cast<int>(i)));
    return results;
}

#include "test_image_ground_truth.hpp"

#include "pnp_verifier.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>

namespace
{
constexpr double kSmallArmorWidthM = 0.135;
constexpr double kSmallArmorHeightM = 0.055;

const std::vector<TestImageGroundTruth> &allTestGroundTruth()
{
    static const std::vector<TestImageGroundTruth> images{
        {"img_1.jpg",
         {{"armor_1", true, {-0.174340, 0.169600, 3.794632}, {-76.991, -51.243, -178.547}, 0},
          {"armor_2", false, {-0.142456, 0.176579, 4.088549}, {-109.737, -48.356, -155.388}, 1},
          {"armor_3", false, {0.150773, 0.162930, 4.055556}, {-107.013, -51.282, -178.516}, 0},
          {"armor_4", true, {0.119301, 0.183937, 3.762585}, {-74.016, -49.423, 158.278}, 1}}},
        {"img_2.jpg",
         {{"armor_1", true, {0.285747, 0.070019, 1.384573}, {-93.155, 64.789, 163.411}, 0},
          {"armor_2", true, {0.009916, 0.097664, 1.281592}, {-74.967, 65.200, -160.239}, 1},
          {"armor_3", false, {-0.094899, 0.047440, 1.553161}, {-123.092, 64.757, 163.479}, 0},
          {"armor_4", false, {0.180498, 0.047525, 1.660025}, {-131.602, 57.188, 136.731}, 1}}},
        {"img_3.jpg",
         {{"armor_1", false, {-0.179673, 0.053321, 1.746903}, {115.614, -61.812, -9.306}, 0},
          {"armor_2", false, {0.102393, 0.059418, 1.835529}, {124.132, -55.870, -36.558}, 1},
          {"armor_3", true, {0.189233, 0.070265, 1.553395}, {85.635, -61.770, -9.327}, 0},
          {"armor_4", true, {-0.092414, 0.092079, 1.467008}, {73.077, -60.945, 21.927}, 1}}}};
    return images;
}

cv::Matx33d rotationFromRpy(const cv::Vec3d &rpy_deg)
{
    const double roll = rpy_deg[0] * CV_PI / 180.0;
    const double pitch = rpy_deg[1] * CV_PI / 180.0;
    const double yaw = rpy_deg[2] * CV_PI / 180.0;
    const cv::Matx33d rx(1.0, 0.0, 0.0,
                         0.0, std::cos(roll), -std::sin(roll),
                         0.0, std::sin(roll), std::cos(roll));
    const cv::Matx33d ry(std::cos(pitch), 0.0, std::sin(pitch),
                         0.0, 1.0, 0.0,
                         -std::sin(pitch), 0.0, std::cos(pitch));
    const cv::Matx33d rz(std::cos(yaw), -std::sin(yaw), 0.0,
                         std::sin(yaw), std::cos(yaw), 0.0,
                         0.0, 0.0, 1.0);
    return rz * ry * rx;
}

cv::Vec3d viewingAngles(const cv::Vec3d &tvec)
{
    return {std::atan2(tvec[0], tvec[2]) * 180.0 / CV_PI,
            std::atan2(tvec[1], tvec[2]) * 180.0 / CV_PI,
            0.0};
}

cv::Point2f imagePointCenter(const std::vector<cv::Point2f> &points)
{
    cv::Point2f center(0.0f, 0.0f);
    for (const cv::Point2f &point : points)
        center += point;
    return points.empty() ? center : center * (1.0f / static_cast<float>(points.size()));
}

cv::Rect2f pointsBounds(const std::vector<cv::Point2f> &points)
{
    if (points.empty())
        return {};
    float min_x = points.front().x;
    float max_x = min_x;
    float min_y = points.front().y;
    float max_y = min_y;
    for (const cv::Point2f &point : points)
    {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }
    return {min_x, min_y, max_x - min_x, max_y - min_y};
}

PoseGroundTruth asPoseGroundTruth(const TestArmorGroundTruth &truth)
{
    PoseGroundTruth pose;
    pose.has_tvec = true;
    pose.tvec = truth.tvec_m;
    pose.has_distance = true;
    pose.distance_m = cv::norm(truth.tvec_m);
    const cv::Vec3d angles = viewingAngles(truth.tvec_m);
    pose.has_yaw = true;
    pose.yaw_deg = angles[0];
    pose.has_pitch = true;
    pose.pitch_deg = angles[1];
    return pose;
}
} // namespace

const TestImageGroundTruth *findTestImageGroundTruth(const std::string &image_path)
{
    const std::string filename = std::filesystem::path(image_path).filename().string();
    const auto &images = allTestGroundTruth();
    const auto found = std::find_if(images.begin(), images.end(), [&filename](const auto &item) {
        return item.image_filename == filename;
    });
    return found == images.end() ? nullptr : &*found;
}

std::vector<ProjectedTruthArmor> projectTestImageGroundTruth(
    const TestImageGroundTruth &truth, const CameraParams &camera_params)
{
    std::vector<ProjectedTruthArmor> projected;
    projected.reserve(truth.armors.size());
    for (const TestArmorGroundTruth &armor : truth.armors)
    {
        const cv::Matx33d rotation = rotationFromRpy(armor.rpy_deg);
        cv::Mat rotation_mat(rotation);
        cv::Mat rvec;
        cv::Rodrigues(rotation_mat, rvec);
        const cv::Mat tvec = (cv::Mat_<double>(3, 1) << armor.tvec_m[0], armor.tvec_m[1], armor.tvec_m[2]);

        const float half_width = static_cast<float>(kSmallArmorWidthM / 2.0);
        const float half_height = static_cast<float>(kSmallArmorHeightM / 2.0);
        std::vector<cv::Point3f> object_points;
        if (armor.width_axis == 0)
        {
            object_points = {{-half_width, 0.0f, -half_height},
                             {half_width, 0.0f, -half_height},
                             {half_width, 0.0f, half_height},
                             {-half_width, 0.0f, half_height}};
        }
        else
        {
            object_points = {{0.0f, -half_width, -half_height},
                             {0.0f, half_width, -half_height},
                             {0.0f, half_width, half_height},
                             {0.0f, -half_width, half_height}};
        }

        ProjectedTruthArmor item;
        item.truth = armor;
        cv::projectPoints(object_points, rvec, tvec, camera_params.camera_matrix,
                          camera_params.dist_coeffs, item.corners);
        std::vector<cv::Point2f> origin_projection;
        cv::projectPoints(std::vector<cv::Point3f>{{0.0f, 0.0f, 0.0f}}, rvec, tvec,
                          camera_params.camera_matrix, camera_params.dist_coeffs,
                          origin_projection);
        if (!origin_projection.empty())
            item.center = origin_projection.front();
        projected.push_back(std::move(item));
    }
    return projected;
}

void drawTestImageGroundTruth(cv::Mat &image,
                              const std::vector<ProjectedTruthArmor> &projected)
{
    static const std::array<cv::Scalar, 4> colors{{
        cv::Scalar(255, 0, 255), cv::Scalar(0, 200, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(80, 255, 80)}};
    for (std::size_t i = 0; i < projected.size(); ++i)
    {
        const ProjectedTruthArmor &armor = projected[i];
        std::vector<cv::Point> corners;
        corners.reserve(armor.corners.size());
        for (const cv::Point2f &point : armor.corners)
            corners.emplace_back(cvRound(point.x), cvRound(point.y));
        if (corners.size() == 4)
            cv::polylines(image, std::vector<std::vector<cv::Point>>{corners}, true,
                          colors[i % colors.size()], 2, cv::LINE_AA);
        cv::drawMarker(image, armor.center, colors[i % colors.size()], cv::MARKER_CROSS, 10, 2, cv::LINE_AA);

        const std::string label = armor.truth.name +
                                  (armor.truth.front_visible ? " front" : " back");
        const cv::Point origin(12, 22 + static_cast<int>(i) * 20);
        cv::putText(image, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(image, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    colors[i % colors.size()], 1, cv::LINE_AA);
    }
}

void printTestImageCalibration(const TestImageGroundTruth &truth,
                               const std::vector<ProjectedTruthArmor> &projected,
                               const CameraParams &camera_params,
                               const cv::Size &image_size)
{
    std::cout << "真值投影标定检查 (" << truth.image_filename << "): " << image_size.width
              << "x" << image_size.height << " px; fx="
              << camera_params.camera_matrix.at<double>(0, 0) << " fy="
              << camera_params.camera_matrix.at<double>(1, 1) << " cx="
              << camera_params.camera_matrix.at<double>(0, 2) << " cy="
              << camera_params.camera_matrix.at<double>(1, 2) << std::endl;
    std::cout << "  假设: tf 相机光学系 XYZ=(右,下,前)，RPY 按 Rz*Ry*Rx；板尺寸采用小板 135x55 mm。"
              << "md 指定宽度轴按板区分 X/Y，高度轴为 Z。" << std::endl;
    std::cout << "  标定重投影 RMS=" << camera_params.calibration_reprojection_error
              << " px (相机标定记录值，不是本次装甲板 PnP 残差)。" << std::endl;

    for (const ProjectedTruthArmor &armor : projected)
    {
        const cv::Rect2f bounds = pointsBounds(armor.corners);
        const bool inside = armor.center.x >= 0.0f && armor.center.x < image_size.width &&
                            armor.center.y >= 0.0f && armor.center.y < image_size.height;
        const cv::Vec3d bearing = viewingAngles(armor.truth.tvec_m);
        std::cout << "  " << armor.truth.name << " ["
                  << (armor.truth.front_visible ? "正面" : "背面") << ", width-axis="
                  << (armor.truth.width_axis == 0 ? "X" : "Y") << "]: center_px=("
                  << std::fixed << std::setprecision(1) << armor.center.x << ", " << armor.center.y
                  << ") bbox_px=(" << bounds.x << "," << bounds.y << "," << bounds.width
                  << "," << bounds.height << ") t_true=(" << std::setprecision(3)
                  << armor.truth.tvec_m[0] << ","
                  << armor.truth.tvec_m[1] << "," << armor.truth.tvec_m[2] << ")m range="
                  << cv::norm(armor.truth.tvec_m) << "m bearing(yaw,pitch)=("
                  << std::setprecision(2) << bearing[0] << "," << bearing[1] << ")deg "
                  << (inside ? "center-in-frame" : "center-outside-frame")
                  << std::defaultfloat << std::setprecision(6) << std::endl;
    }
}

void printTestImagePoseErrors(const std::vector<ProjectedTruthArmor> &projected,
                              const std::vector<PnPResult> &estimates)
{
    std::vector<const PnPResult *> valid_estimates;
    for (const PnPResult &estimate : estimates)
        if (estimate.valid)
            valid_estimates.push_back(&estimate);

    std::cout << "真值误差分析: " << valid_estimates.size() << " 个有效 PnP 解 / "
              << projected.size() << " 块真值板。" << std::endl;
    if (valid_estimates.empty())
    {
        std::cout << "  当前无有效检测/PnP，无法计算位姿误差；真值轮廓仍已投影绘制。" << std::endl;
        return;
    }

    std::vector<bool> truth_used(projected.size(), false);
    for (const PnPResult *estimate : valid_estimates)
    {
        const cv::Point2f estimate_center = imagePointCenter(estimate->image_points);
        std::size_t best_index = projected.size();
        double best_distance = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < projected.size(); ++i)
        {
            if (truth_used[i])
                continue;
            const double distance = cv::norm(estimate_center - projected[i].center);
            if (distance < best_distance)
            {
                best_distance = distance;
                best_index = i;
            }
        }
        if (best_index == projected.size())
            continue;
        truth_used[best_index] = true;

        const TestArmorGroundTruth &truth = projected[best_index].truth;
        const PoseError error = compareToGroundTruth(*estimate, asPoseGroundTruth(truth));
        std::cout << "  Armor " << (estimate->armor_index + 1) << " -> " << truth.name
                  << " center_delta=" << best_distance << "px";
        if (error.has_tvec_error)
        {
            std::cout << " delta_t=(" << error.tvec_error[0] << "," << error.tvec_error[1]
                      << "," << error.tvec_error[2] << ")m |delta_t|="
                      << cv::norm(error.tvec_error) << "m";
        }
        if (error.has_distance_error)
            std::cout << " delta_range=" << error.distance_error_m << "m";
        if (error.has_yaw_error)
            std::cout << " delta_bearing_yaw=" << error.yaw_error_deg << "deg";
        if (error.has_pitch_error)
            std::cout << " delta_bearing_pitch=" << error.pitch_error_deg << "deg";
        std::cout << " reprojection_RMS=" << estimate->reprojection_error_px << "px"
                  << std::defaultfloat << std::setprecision(6) << std::endl;
    }
}

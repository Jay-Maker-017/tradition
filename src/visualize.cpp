#include "visualize.hpp"

#include <algorithm>
#include <string>

// 可视化模块

namespace
{
// 是否绘制灯条红蓝矩形框；绿色中心点不受此开关影响。
constexpr bool kDrawLightbarRects = false;
constexpr bool kDrawPnPResults = true;
} // namespace

void drawLightbars(cv::Mat &img, const std::vector<Lightbar> &lightbars,
                   bool draw_rects)
{
    for (const auto &bar : lightbars)
    {
        if (draw_rects)
        {
            // 蓝灯条画蓝框，红灯条画红框
            const cv::Scalar color = bar.is_blue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);

            // 绘制旋转矩形
            cv::Point2f vertices[4];
            bar.rect.points(vertices);
            for (int i = 0; i < 4; i++)
                cv::line(img, vertices[i], vertices[(i + 1) % 4], color, 3, cv::LINE_AA);
        }

        cv::circle(img, bar.rect.center, 3, cv::Scalar(0, 255, 0), -1); // 中心点
    }
}

void drawArmors(cv::Mat &img, const std::vector<Armor> &armors)
{
    if (armors.empty())
    {
        // 没有配对成功的装甲板时给个文字提示，便于确认这一环节确实跑了
        cv::putText(img, "No armor detected", cv::Point(15, img.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        return;
    }

    // 绿色：装甲板轮廓
    const cv::Scalar armor_color(0, 255, 0);
    for (size_t k = 0; k < armors.size(); ++k)
    {
        // 灯条配对成功但数字神经网络未识别出类型的装甲板不绘制：
        // 不画绿色边框、不画中心点、不画标签文字。
        if (!armors[k].has_type)
            continue;

        // 四个顶点：[左上, 右上, 右下, 左下]
        const std::vector<cv::Point2f> corners = armorCorners(armors[k].left, armors[k].right);

        std::vector<cv::Point> int_corners;
        int_corners.reserve(corners.size());
        for (const auto &p : corners)
            int_corners.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));

        // 画闭合四边形
        cv::polylines(img, int_corners, /*isClosed=*/true, armor_color, 2, cv::LINE_AA);

        // 画板中心实心点 + 编号文字
        cv::Point2f center(0.f, 0.f);
        float top_y = corners[0].y; // 四边形最高点 y，用于把文字放在框上方
        for (const auto &p : corners)
        {
            center += p;
            top_y = std::min(top_y, p.y);
        }
        center *= 0.25f;
        cv::circle(img, center, 4, armor_color, cv::FILLED, cv::LINE_AA);

        // 标签：编号 + 数字神经网络识别出的类型（能走到这里说明已识别出类型）
        const std::string label = "Armor " + std::to_string(k + 1) + ": " +
                                  armorTypeNameEn(armors[k].armor_id) + " " +
                                  cv::format("%.2f", armors[k].nn_score);

        int base_line = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base_line);
        // 文字放在四边形上方水平居中，并保证不超出图像上边界
        cv::Point org(static_cast<int>(center.x - text_size.width / 2.f),
                      std::max(text_size.height + 4, static_cast<int>(top_y) - 8));
        // 识别到类型时用黄色标注
        const cv::Scalar text_color(0, 255, 255);
        cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv::LINE_AA);
    }
}

cv::Mat makeAnnotatedFrame(const cv::Mat &frame,
                           const std::vector<Armor> &armors)
{
    cv::Mat result = frame.clone();

    // 只有配对成功且识别出类型的装甲板，其灯条才绘制中心绿点
    std::vector<Lightbar> confirmed_bars;
    confirmed_bars.reserve(armors.size() * 2);
    for (const Armor &armor : armors)
    {
        if (!armor.has_type)
            continue;
        confirmed_bars.push_back(armor.left);
        confirmed_bars.push_back(armor.right);
    }

    drawLightbars(result, confirmed_bars, kDrawLightbarRects);
    drawArmors(result, armors);
    return result;
}

cv::Mat resizeForDisplay(const cv::Mat &image)
{
    constexpr int min_display_width = 460;
    constexpr int max_display_width = 1280;
    cv::Mat display = image;
    if (display.cols < min_display_width || display.cols > max_display_width)
    {
        const int target_width = std::clamp(display.cols, min_display_width, max_display_width);
        const double scale = static_cast<double>(target_width) / display.cols;
        cv::resize(image, display, cv::Size(), scale, scale);
    }
    return display;
}

void drawPnPResults(cv::Mat &img, const std::vector<PnPResult> &results,
                    const CameraParams &camera_params)
{
    if (!kDrawPnPResults)
        return;

    for (const PnPResult &result : results)
    {
        if (!result.valid)
            continue;

        for (const cv::Point2f &point : result.reprojected_points)
            cv::circle(img, point, 5, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);

        cv::drawFrameAxes(img, camera_params.camera_matrix, camera_params.dist_coeffs,
                          result.rvec, result.tvec, 0.1f, 2);

        cv::Point2f center(0.f, 0.f);
        if (!result.reprojected_points.empty())
        {
            for (const cv::Point2f &point : result.reprojected_points)
                center += point;
            center *= 1.f / static_cast<float>(result.reprojected_points.size());
        }
        else if (!result.image_points.empty())
        {
            for (const cv::Point2f &point : result.image_points)
                center += point;
            center *= 1.f / static_cast<float>(result.image_points.size());
        }

        const std::string distance_label = "D=" + cv::format("%.2f", result.distance_m) + " m";
        const cv::Point label_origin(cvRound(center.x + 8.f), cvRound(center.y + 22.f));
        cv::putText(img, distance_label, label_origin, cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    }
}

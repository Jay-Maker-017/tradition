#include "visualize.hpp"

#include <algorithm>
#include <string>

// ==================== 可视化模块实现 ====================

void drawLightbars(cv::Mat &img, const std::vector<Lightbar> &lightbars)
{
    for (const auto &bar : lightbars)
    {
        // 蓝灯条画蓝框，红灯条画红框
        const cv::Scalar color = bar.is_blue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);

        // 绘制旋转矩形
        cv::Point2f vertices[4];
        bar.rect.points(vertices);
        for (int i = 0; i < 4; i++)
            cv::line(img, vertices[i], vertices[(i + 1) % 4], color, 3, cv::LINE_AA);

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

    // 绿色：装甲板轮廓（与单灯条的红/蓝框区分开）
    const cv::Scalar armor_color(0, 255, 0);
    for (size_t k = 0; k < armors.size(); ++k)
    {
        // 四个顶点：[左上, 右上, 右下, 左下]（与调用方打印的坐标完全一致）
        const std::vector<cv::Point2f> corners = armorCorners(armors[k].left, armors[k].right);

        // polylines 要求整数点集（CV_32S），先四舍五入转换
        std::vector<cv::Point> int_corners;
        int_corners.reserve(corners.size());
        for (const auto &p : corners)
            int_corners.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));

        // 用 polylines 一次画闭合四边形（isClosed=true 自动首尾相连）
        cv::polylines(img, int_corners, /*isClosed=*/true, armor_color, 2, cv::LINE_AA);

        // 板中心 = 四个顶点的形心（均值），画实心点 + 编号文字
        cv::Point2f center(0.f, 0.f);
        float top_y = corners[0].y; // 四边形最高点 y，用于把文字放在框上方
        for (const auto &p : corners)
        {
            center += p;
            top_y = std::min(top_y, p.y);
        }
        center *= 0.25f;
        cv::circle(img, center, 4, armor_color, cv::FILLED, cv::LINE_AA);

        const std::string label = "Armor " + std::to_string(k + 1);
        int base_line = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base_line);
        // 文字放在四边形上方水平居中，并保证不超出图像上边界
        cv::Point org(static_cast<int>(center.x - text_size.width / 2.f),
                      std::max(text_size.height + 4, static_cast<int>(top_y) - 8));
        cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, armor_color, 2, cv::LINE_AA);
    }
}

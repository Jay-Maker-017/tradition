#include "lightbar.hpp"

#include <algorithm>
#include <cmath>

// ==================== 灯条检测模块实现 ====================

std::vector<Lightbar> LightbarDetector::detect(const cv::Mat &bgr_img) const
{
    std::vector<Lightbar> lightbars;

    // 转灰度
    cv::Mat gray_img;
    cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

    // 二值化（灯条是高亮区域，取大于阈值的像素）
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, params_.threshold, 255, cv::THRESH_BINARY);

    // 找外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 逐个轮廓尝试构造灯条
    for (const auto &contour : contours)
    {
        Lightbar bar;
        if (tryMakeLightbar(bgr_img, contour, static_cast<int>(lightbars.size()), bar))
            lightbars.push_back(bar);
    }
    return lightbars;
}

bool LightbarDetector::tryMakeLightbar(const cv::Mat &bgr_img,
                                       const std::vector<cv::Point> &contour,
                                       int id, Lightbar &out) const
{
    if (static_cast<int>(contour.size()) < params_.min_contour_size)
        return false; // 将太小的轮廓忽略

    const cv::RotatedRect rect = cv::minAreaRect(contour);

    // 计算长宽（长边恒为 length，短边恒为 width）
    const float width = std::min(rect.size.width, rect.size.height);
    const float height = std::max(rect.size.width, rect.size.height);
    if (width <= 0.f)
        return false; // 防止除零

    const float ratio = height / width; // 长宽比恒大于 1

    // 几何筛选
    if (!(ratio > params_.min_ratio && ratio < params_.max_ratio &&
          height > params_.min_length))
        return false;

    // 颜色判别：蓝通道均值更大即为蓝灯条
    const bool is_blue = classifyIsBlue(bgr_img, contour);

    out = makeLightbar(rect, is_blue, id);
    return true;
}

bool LightbarDetector::classifyIsBlue(const cv::Mat &bgr_img,
                                      const std::vector<cv::Point> &contour) const
{
    // 用灯条区域内所有像素的均值比较蓝/红通道，比只看轮廓边界点更稳定
    cv::Mat mask = cv::Mat::zeros(bgr_img.size(), CV_8UC1);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0,
                     cv::Scalar(255), cv::FILLED);
    const cv::Scalar mean = cv::mean(bgr_img, mask);
    return mean[0] > mean[2]; // B > R → 蓝色
}

Lightbar LightbarDetector::makeLightbar(const cv::RotatedRect &rect, bool is_blue, int id)
{
    Lightbar bar;
    bar.rect = rect;
    bar.center = rect.center;
    bar.length = std::max(rect.size.width, rect.size.height);
    bar.width = std::min(rect.size.width, rect.size.height);
    bar.is_blue = is_blue;
    bar.id = id;

    // 取出四个角点，找出最长边（即长轴方向）
    cv::Point2f pts[4];
    rect.points(pts);
    cv::Point2f best(0.f, 0.f);
    float best_len = 0.f;
    for (int i = 0; i < 4; ++i)
    {
        const cv::Point2f d = pts[(i + 1) % 4] - pts[i];
        const float len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len > best_len)
        {
            best_len = len;
            best = d;
        }
    }
    // 归一化成单位方向向量；方向正负无所谓（配对判据都取绝对值）
    if (best_len > 1e-6f)
        bar.axis = best / best_len;
    else
        bar.axis = cv::Point2f(0.f, 1.f);
    return bar;
}

#include "digit_recognizer.hpp"

#include <cstddef>
#include <limits>

// 负责神经网络一路的装甲板匹配


namespace
{
// 计算网络装甲板四角点的中心（points 顺序：左上、左下、右下、右上）
cv::Point2f netCenter(const NetArmorResult &result)
{
    cv::Point2f center(0.f, 0.f);
    if (result.points.empty())
        return center;
    for (const cv::Point2d &p : result.points)
        center += cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
    return center * (1.f / static_cast<float>(result.points.size()));
}
} // namespace

DigitRecognizer::DigitRecognizer(const DigitParams &params)
    : params_(params),
      model_(JsonConfig{params.json_path, params.model_key, params.model_folder})
{
}

std::vector<NetArmorResult> DigitRecognizer::detect(const cv::Mat &bgr_img)
{
    if (bgr_img.empty())
        return {};
    return model_.netProcess(bgr_img, params_.my_color);
}

void DigitRecognizer::recognize(const cv::Mat &bgr_img, std::vector<Armor> &armors)
{
    const std::vector<NetArmorResult> results = detect(bgr_img);
    if (results.empty())
        return;

    // 网络结果一对一使用，避免多块传统装甲板匹配到同一个网络目标
    std::vector<bool> used(results.size(), false);

    for (Armor &armor : armors)
    {
        const cv::Point2f center = armorCenter(armor.left, armor.right);
        const float width = cv::norm(armor.right.center - armor.left.center);
        if (width < 1e-3f)
            continue;

        // 找中心距最近的网络装甲板
        int best = -1;
        float best_dist = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            if (used[i] || results[i].points.size() != 4)
                continue;
            const float d = cv::norm(netCenter(results[i]) - center);
            if (d < best_dist)
            {
                best_dist = d;
                best = static_cast<int>(i);
            }
        }

        // 距离在允许范围内才认为是同一块装甲板，把类型写回传统结果
        if (best >= 0 && best_dist <= params_.match_dist_ratio * width)
        {
            used[best] = true;
            const NetArmorResult &r = results[best];
            armor.has_type = true;
            armor.armor_id = r.armor_id;
            armor.color_id = r.color_id;
            armor.nn_score = r.score;
            armor.class_name = r.class_name;
        }
    }
}

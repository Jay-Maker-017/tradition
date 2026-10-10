#include "armor.hpp"

#include <algorithm>
#include <cmath>

// 装甲板配对模块实现思路:

// 提取单根灯条长轴方向的两个中点作为两个端点
static void barEndpoints(const Lightbar &bar, const cv::Point2f &perp,
                         cv::Point2f &up, cv::Point2f &down)
{
    cv::Point2f pts[4];
    bar.rect.points(pts);
    // 按在 perp 上的投影值升序排列（投影值 = 角点与 perp 的点积）
    std::sort(pts, pts + 4, [&perp](const cv::Point2f &a, const cv::Point2f &b) {
        return (a.x * perp.x + a.y * perp.y) < (b.x * perp.x + b.y * perp.y);
    });
    down = 0.5f * (pts[0] + pts[1]); // 投影最小的两个角点 -> 下端中点
    up = 0.5f * (pts[2] + pts[3]);   // 投影最大的两个角点 -> 上端中点
}

std::vector<cv::Point2f> armorCorners(const Lightbar &left, const Lightbar &right)
{
    // 两灯条中心连线的方向向量及其垂线
    const cv::Point2f dc = right.center - left.center;
    const float dist = std::max(std::sqrt(dc.x * dc.x + dc.y * dc.y), 1e-6f); // 防除零
    const cv::Point2f dir = dc / dist; 
    cv::Point2f perp(-dir.y, dir.x);   // 与连线垂直的单位向量
    if (perp.y > 0.f)
        perp = -perp; // 让 perp 指向y 减小方向，保证"上端"确为"视觉上的上方"

    cv::Point2f lt, lb, rt, rb;
    barEndpoints(left, perp, lt, lb);  // 左灯条的上、下端
    barEndpoints(right, perp, rt, rb); // 右灯条的上、下端

    return {lt, rt, rb, lb}; // 左上 -> 右上 -> 右下 -> 左下
}

float ArmorMatcher::pairScore(const Lightbar &a, const Lightbar &b) const
{
    // 判据 1：颜色一致
    if (a.is_blue != b.is_blue)
        return -1.f;

    // 判据 2：长度接近
    const float max_len = std::max(a.length, b.length);
    if (max_len <= 2e-6f)
        return -1.f;
    const float len_diff = std::fabs(a.length - b.length) / max_len;
    if (len_diff > params_.max_len_diff_ratio)
        return -1.f;

    // 判据 3：距离合理 (中心距与平均长度之比要落在合理区间)
    const cv::Point2f dc = b.center - a.center;
    const float dist = std::sqrt(dc.x * dc.x + dc.y * dc.y);
    const float avg_len = 0.5f * (a.length + b.length);
    if (avg_len <= 2e-6f)
        return -1.f;
    const float dist_ratio = dist / avg_len;
    if (dist_ratio < params_.min_dist_ratio || dist_ratio > params_.max_dist_ratio)
        return -1.f;

    // 判据 4：斜率接近
    const float cross = std::fabs(a.axis.x * b.axis.y - a.axis.y * b.axis.x);
    if (cross > params_.max_parallel_err)
        return -1.f;

    // 判据 5："并排"检测 (两灯条中心连线要近似垂直于长轴)
    const cv::Point2f dir = dc / std::max(dist, 2e-6f); // 连线方向单位向量
    const float dot = std::fabs(dir.x * a.axis.x + dir.y * a.axis.y);
    if (dot > params_.max_perp_err)
        return -1.f;

    // 对全部硬判据计算加权得分，越小越优
    const float dist_err = std::fabs(dist_ratio - params_.ideal_dist_ratio) / params_.ideal_dist_ratio;
    return params_.w_len_diff * len_diff + params_.w_parallel * cross +
           params_.w_perp * dot + params_.w_dist * dist_err;
}

std::vector<Armor> ArmorMatcher::match(const std::vector<Lightbar> &bars) const
{
    // 候选对：{得分, 灯条 i, 灯条 j}
    struct Candidate
    {
        float score;
        int i;
        int j;
    };
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < bars.size(); ++i)
    {
        for (size_t j = i + 1; j < bars.size(); ++j)
        {
            const float s = pairScore(bars[i], bars[j]);
            if (s >= 0.f)
                candidates.push_back({s, static_cast<int>(i), static_cast<int>(j)});
        }
    }

    // 得分升序
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.score < b.score; });

    std::vector<bool> used(bars.size(), false);
    std::vector<Armor> armors;
    for (const auto &c : candidates)
    {
        if (used[c.i] || used[c.j])
            continue; // 保证一对一
        used[c.i] = true;
        used[c.j] = true;
        // 统一按 x 坐标摆放：左灯条在前、右灯条在后
        Armor armor;
        if (bars[c.i].center.x <= bars[c.j].center.x)
            armor = {bars[c.i], bars[c.j]};
        else
            armor = {bars[c.j], bars[c.i]};
        armors.push_back(armor);
    }

    // 按装甲板中心 x 坐标从左到右排序
    std::sort(armors.begin(), armors.end(), [](const Armor &a, const Armor &b) {
        return 0.5f * (a.left.center.x + a.right.center.x) <
               0.5f * (b.left.center.x + b.right.center.x);
    });
    return armors;
}

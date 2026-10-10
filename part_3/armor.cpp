#include "armor.hpp"

#include <algorithm>
#include <cmath>

// ==================== 装甲板配对模块实现 ====================

// 提取单根灯条长轴方向的两个端点（端面中点）：
// 把 4 个角点投影到"两灯条中心连线的垂线"perp 上，
// 投影最大的两个角点属于同一端（up），最小的两个属于另一端（down）；
// 每端取两角点的中点作为端点 → 四边形宽边穿过灯条中心线，左右对称。
// 注意：不直接用 Lightbar::axis 定端（其正负取决于角点遍历顺序、并不固定），
//       用 perp 投影判定才能保证两根灯条的上下端严格一一对应，避免四边形交叉成蝴蝶结。
static void barEndpoints(const Lightbar &bar, const cv::Point2f &perp,
                         cv::Point2f &up, cv::Point2f &down)
{
    cv::Point2f pts[4];
    bar.rect.points(pts);
    // 按在 perp 上的投影值升序排列（投影值 = 角点与 perp 的点积）
    std::sort(pts, pts + 4, [&perp](const cv::Point2f &a, const cv::Point2f &b) {
        return (a.x * perp.x + a.y * perp.y) < (b.x * perp.x + b.y * perp.y);
    });
    down = 0.5f * (pts[0] + pts[1]); // 投影最小的两个角点 → 下端中点
    up = 0.5f * (pts[2] + pts[3]);   // 投影最大的两个角点 → 上端中点
}

std::vector<cv::Point2f> armorCorners(const Lightbar &left, const Lightbar &right)
{
    // 两灯条中心连线的方向向量及其垂线
    const cv::Point2f dc = right.center - left.center;
    const float dist = std::max(std::sqrt(dc.x * dc.x + dc.y * dc.y), 1e-6f); // 防除零
    const cv::Point2f dir = dc / dist; // 连线单位向量（左灯条 → 右灯条）
    cv::Point2f perp(-dir.y, dir.x);   // 与连线垂直的单位向量
    if (perp.y > 0.f)
        perp = -perp; // 让 perp 指向图像上方（y 减小方向），保证"上端"确为视觉上的上方

    cv::Point2f lt, lb, rt, rb;
    barEndpoints(left, perp, lt, lb);  // 左灯条的上、下端
    barEndpoints(right, perp, rt, rb); // 右灯条的上、下端

    return {lt, rt, rb, lb}; // 左上 → 右上 → 右下 → 左下，闭合即四边形
}

float ArmorMatcher::pairScore(const Lightbar &a, const Lightbar &b) const
{
    // 判据 1：颜色必须一致（两蓝或两红才能组成一块板）
    if (a.is_blue != b.is_blue)
        return -1.f;

    // 判据 2：两灯条长度要接近（长度差 / 较长者）
    const float max_len = std::max(a.length, b.length);
    if (max_len <= 1e-6f)
        return -1.f;
    const float len_diff = std::fabs(a.length - b.length) / max_len;
    if (len_diff > params_.max_len_diff_ratio)
        return -1.f;

    // 判据 3：中心距与平均长度之比要落在合理区间（太近/太远都不是一块板）
    const cv::Point2f dc = b.center - a.center;
    const float dist = std::sqrt(dc.x * dc.x + dc.y * dc.y);
    const float avg_len = 0.5f * (a.length + b.length);
    if (avg_len <= 1e-6f)
        return -1.f;
    const float dist_ratio = dist / avg_len;
    if (dist_ratio < params_.min_dist_ratio || dist_ratio > params_.max_dist_ratio)
        return -1.f;

    // 判据 4：两长轴方向要接近平行
    //   cross = |sin(夹角)|，0 表示完全平行，1 表示完全垂直（方向正负不影响，故取绝对值）
    const float cross = std::fabs(a.axis.x * b.axis.y - a.axis.y * b.axis.x);
    if (cross > params_.max_parallel_err)
        return -1.f;

    // 判据 5：两灯条中心连线要近似垂直于长轴（即"并排"而不是"首尾相接"）
    const cv::Point2f dir = dc / std::max(dist, 1e-6f); // 连线方向单位向量
    const float dot = std::fabs(dir.x * a.axis.x + dir.y * a.axis.y);
    if (dot > params_.max_perp_err)
        return -1.f;

    // 全部硬判据通过 → 计算加权得分
    // 各项误差都归一化到相近量级（0~1），加权求和，越小越优
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

    // 得分升序：越优（越小）的候选对越先被接受
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.score < b.score; });

    std::vector<bool> used(bars.size(), false);
    std::vector<Armor> armors;
    for (const auto &c : candidates)
    {
        if (used[c.i] || used[c.j])
            continue; // 任意一根已被占用则跳过，保证一对一
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

    // 最后按装甲板中心 x 坐标从左到右排序，保证控制台打印与图上 "Armor N" 编号一致
    std::sort(armors.begin(), armors.end(), [](const Armor &a, const Armor &b) {
        return 0.5f * (a.left.center.x + a.right.center.x) <
               0.5f * (b.left.center.x + b.right.center.x);
    });
    return armors;
}

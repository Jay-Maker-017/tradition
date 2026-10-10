#pragma once

#include "lightbar.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

// ==================== 装甲板配对模块：数据结构与配对器 ====================

// 一块装甲板 = 配对成功的左右两根灯条（left 在左、right 在右，按 x 坐标排好）
struct Armor
{
    Lightbar left;
    Lightbar right;
};

// 配对参数（集中在此处，便于按不同图片微调；默认值即原单文件版本的取值）
struct ArmorParams
{
    float max_len_diff_ratio = 0.35f; // 长度差/较长者 的允许上限（两灯条长度需接近）
    float min_dist_ratio = 0.8f;      // 中心距/平均长度 的下限（太近不算一块板）
    float max_dist_ratio = 4.5f;      // 中心距/平均长度 的上限（太远不算一块板）
    float ideal_dist_ratio = 2.6f;    // 理想间距比（按实测测试图 ≈2.6~2.8 取值，用于打分，越接近越优）
    float max_parallel_err = 0.2f;    // 平行误差上限（对应两长轴夹角约 11.5°）
    float max_perp_err = 0.35f;       // 垂直误差上限（对应中心连线与长轴夹角约 20°）
    // 得分项权重（默认均为 1.0，按需调整可改变各项判据的话语权）
    float w_len_diff = 1.0f;          // 长度差
    float w_parallel = 1.0f;          // 平行误差
    float w_perp = 1.0f;              // 垂直误差
    float w_dist = 1.0f;              // 间距偏离理想值
};

// 装甲板配对器：输入若干灯条，输出配对成功的装甲板列表。
// 只做几何配对，不画图、不打印，便于直接移植到其他工程。
class ArmorMatcher
{
public:
    ArmorMatcher() = default;
    explicit ArmorMatcher(const ArmorParams &params) : params_(params) {}

    // 设置/获取配对参数
    void setParams(const ArmorParams &params) { params_ = params; }
    const ArmorParams &params() const { return params_; }

    // 配对主入口：所有候选对按得分从优到劣贪心一对一，每根灯条最多属于一块装甲板
    std::vector<Armor> match(const std::vector<Lightbar> &bars) const;

    // 计算两根灯条构成装甲板的得分：返回 -1 表示不满足硬判据；否则得分越小越优。
    // 公开此方法是为了让调用方打印得分时与配对内部使用完全一致的参数。
    float pairScore(const Lightbar &a, const Lightbar &b) const;

private:
    ArmorParams params_; // 配对参数
};

// 计算一块装甲板的 4 个角点（四边形顶点）：
// 每根灯条取长轴方向的两个端点，共 4 个点；
// 返回顺序固定为 [左上, 右上, 右下, 左下]，
// 供调用方打印与绘图共用，保证"打印的坐标"就是"图上画的顶点"。
std::vector<cv::Point2f> armorCorners(const Lightbar &left, const Lightbar &right);

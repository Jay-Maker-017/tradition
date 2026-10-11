#pragma once

#include "lightbar.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 装甲板配对模块
// 一块装甲板 = 两根灯条 left 在左、right 在右，按 x 坐标排好
struct Armor
{
    Lightbar left;
    Lightbar right;

    // 数字神经网络识别结果
    bool has_type = false;  // 是否成功识别出装甲板类型
    int armor_id = -1;      // 类型 ID：0哨兵 1英雄 2工程 3/4/5号步兵 6前哨站 7基地(底部) 8基地(顶部)
    int color_id = -1;      // 颜色 ID：0蓝 1红 2白 3紫
    int size = -1;          // 装甲板尺寸：0小板 1大板 -1未知
    double nn_score = 0.;   // 网络置信度
    std::string class_name; // 类型名称
};

// 配对参数
struct ArmorParams
{
    float max_len_diff_ratio = 0.35f; // 两灯条长度接近度 (长度差/较长者 的允许上限)
    float min_dist_ratio = 0.8f;      // 中心距/平均长度 的下限
    float max_dist_ratio = 4.5f;      // 中心距/平均长度 的上限
    float ideal_dist_ratio = 2.6f;    // 理想间距比（实测一般 2.6~2.8 取值，用于打分，越接近越优）
    float max_parallel_err = 0.2f;    // 平行误差上限（对应两长轴夹角约 11.5°）
    float max_perp_err = 0.35f;       // 垂直误差上限（对应中心连线与长轴夹角约 20°）
    // 得分项权重
    float w_len_diff = 1.0f;          // 长度差
    float w_parallel = 1.0f;          // 平行误差
    float w_perp = 1.0f;              // 垂直误差
    float w_dist = 1.0f;              // 间距偏离理想值
};

// 装甲板配对器：输入若干灯条，输出配对成功的装甲板列表
class ArmorMatcher
{
public:
    ArmorMatcher() = default;
    explicit ArmorMatcher(const ArmorParams &params) : params_(params) {}

    // 设置配对参数
    void setParams(const ArmorParams &params) { params_ = params; }
    const ArmorParams &params() const { return params_; }

    // 配对主入口
    std::vector<Armor> match(const std::vector<Lightbar> &bars) const;

    // 计算两根灯条构成装甲板的得分,越小越优
    float pairScore(const Lightbar &a, const Lightbar &b) const;

private:
    ArmorParams params_; // 配对参数
};

// 计算一块装甲板的 4 个角点 ：
std::vector<cv::Point2f> armorCorners(const Lightbar &left, const Lightbar &right);

// 计算一块装甲板的中心（两灯条中心的中点）
cv::Point2f armorCenter(const Lightbar &left, const Lightbar &right);

// 类型 ID -> 英文名称
const char *armorTypeNameEn(int armor_id);

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// ==================== 灯条检测模块：数据结构与检测器 ====================

// 单根灯条的描述（由检测器输出，配对模块的输入）
struct Lightbar
{
    cv::RotatedRect rect; // 灯条的旋转矩形（minAreaRect 结果）
    cv::Point2f center;   // 灯条中心
    float length = 0.f;   // 长边长度（灯条沿自身方向的长度）
    float width = 0.f;    // 短边长度（灯条宽度）
    cv::Point2f axis;     // 长轴单位方向向量（由四点边长判定，规避 RotatedRect::angle 的版本差异）
    bool is_blue = false; // 颜色：true=蓝灯条，false=红灯条
    int id = 0;           // 序号（即检出顺序），方便调试与打印
};

// 灯条检测参数（默认值即原课程代码中的取值，可按图片调整）
struct LightbarParams
{
    double threshold = 200.0;      // 二值化阈值（THRESH_BINARY）
    double min_ratio = 3.0;        // 最小长宽比
    double max_ratio = 9.0;        // 最大长宽比
    double min_length = 5.0;       // 灯条最小长度（长边像素数）
    int min_contour_size = 18;     // 轮廓点数下限，太小的轮廓直接忽略
};

// 灯条检测器：输入 BGR 原图，输出若干根候选灯条（含颜色与几何信息）。
// 只做检测，不画图、不打印，便于直接移植到其他工程。
class LightbarDetector
{
public:
    LightbarDetector() = default;
    explicit LightbarDetector(const LightbarParams &params) : params_(params) {}

    // 设置/获取检测参数（例如按不同图片调整阈值）
    void setParams(const LightbarParams &params) { params_ = params; }
    const LightbarParams &params() const { return params_; }

    // 检测主入口：完整流程 = 灰度 → 二值化 → 找轮廓 → 几何筛选 → 颜色判别
    std::vector<Lightbar> detect(const cv::Mat &bgr_img) const;

private:
    // 逐轮廓：判断是否能构成灯条，能则填充 out，返回是否成功
    bool tryMakeLightbar(const cv::Mat &bgr_img, const std::vector<cv::Point> &contour,
                         int id, Lightbar &out) const;

    // 用灯条区域内所有像素的 B/R 通道均值判断颜色（比只看边界点更稳定）
    bool classifyIsBlue(const cv::Mat &bgr_img, const std::vector<cv::Point> &contour) const;

    // 由 minAreaRect 结果构造 Lightbar（长轴方向用四点最长边判定，不依赖 RotatedRect::angle）
    static Lightbar makeLightbar(const cv::RotatedRect &rect, bool is_blue, int id);

    LightbarParams params_; // 检测参数
};

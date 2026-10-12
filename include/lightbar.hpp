#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// 灯条检测模块

// 单根灯条的描述
struct Lightbar
{
    cv::RotatedRect rect; // 灯条的旋转矩形
    cv::Point2f center;   // 灯条中心
    float length = 0.f;   // 长边长度
    float width = 0.f;    // 短边长度
    cv::Point2f axis;     // 长轴单位方向向量
    bool is_blue = false; // 颜色
    int id = 0;           // 序号
};

// 灯条检测参数
struct LightbarParams
{
    double threshold = 200.0;      // 二值化阈值
    double min_ratio = 3.0;        // 最小长宽比
    double max_ratio = 50.0;        // 最大长宽比
    double min_length = 5.0;       // 灯条长边最小像素数
    int min_contour_size = 12;     // 轮廓点数下限
};

// 灯条检测器
class LightbarDetector
{
public:
    LightbarDetector() = default;
    explicit LightbarDetector(const LightbarParams &params) : params_(params) {}

    // 设置检测参数
    void setParams(const LightbarParams &params) { params_ = params; }
    const LightbarParams &params() const { return params_; }

    // 检测主入口
    std::vector<Lightbar> detect(const cv::Mat &bgr_img) const;

private:
    // 判断是否能构成灯条
    bool tryMakeLightbar(const cv::Mat &bgr_img, const std::vector<cv::Point> &contour,
                         int id, Lightbar &out) const;

    // 判断灯条颜色
    bool classifyIsBlue(const cv::Mat &bgr_img, const std::vector<cv::Point> &contour) const;

    
    static Lightbar makeLightbar(const cv::RotatedRect &rect, bool is_blue, int id);

    LightbarParams params_; // 检测参数
};

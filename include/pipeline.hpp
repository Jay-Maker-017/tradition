#pragma once

#include "digit_recognizer.hpp"
#include "lightbar.hpp"

#include <memory>
#include <string>
#include <vector>

// 单帧检测结果。
struct DetectionResult
{
    std::vector<Lightbar> lightbars;
    std::vector<Armor> armors;
};

// 编排传统检测、装甲板配对与可选数字识别的帧处理流水线。
class DetectionPipeline
{
public:
    explicit DetectionPipeline(const DigitParams &digit_params);

    // 处理单帧；数字识别运行时失败后自动关闭后续帧的网络推理。
    DetectionResult processFrame(const cv::Mat &bgr_img);

    bool digitRecognitionEnabled() const { return digit_recognition_enabled_; }
    const std::string &recognitionError() const { return recognition_error_; }
    const ArmorMatcher &matcher() const { return matcher_; }

private:
    LightbarDetector detector_;
    ArmorMatcher matcher_;
    std::unique_ptr<DigitRecognizer> recognizer_;
    bool digit_recognition_enabled_ = false;
    std::string recognition_error_;
};

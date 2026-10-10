#include "pipeline.hpp"

#include <exception>
#include <iostream>

DetectionPipeline::DetectionPipeline(const DigitParams &digit_params)
{
    try
    {
        recognizer_ = std::make_unique<DigitRecognizer>(digit_params);
        digit_recognition_enabled_ = true;
    }
    catch (const std::exception &e)
    {
        recognition_error_ = e.what();
    }
}

DetectionResult DetectionPipeline::processFrame(const cv::Mat &bgr_img)
{
    DetectionResult result;
    result.lightbars = detector_.detect(bgr_img);
    result.armors = matcher_.match(result.lightbars);

    if (recognizer_ != nullptr && digit_recognition_enabled_)
    {
        try
        {
            recognizer_->recognize(bgr_img, result.armors);
        }
        catch (const std::exception &e)
        {
            digit_recognition_enabled_ = false;
            recognition_error_ = e.what();
            std::cerr << "[警告] 数字识别推理失败: " << e.what()
                      << "；后续帧仅运行传统检测。" << std::endl;
        }
    }

    return result;
}

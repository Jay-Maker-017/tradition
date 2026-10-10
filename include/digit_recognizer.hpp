#pragma once

#include "armor.hpp"
#include "network_deployment_interface.hpp"

#include <string>
#include <vector>

// 装甲板类型（数字）神经网络识别模块

// 思路参考 提供数字识别神经网络的那个开源工程：
//     用 NNdeployment 对整帧图像做一次网络推理
//     得到装甲板（四角点 + 类型,颜色,尺寸）
//     再按中心距与传统方案检测出的装甲板逐一匹配，
//     把类型信息补充到传统 Armor 上

// 数字识别参数
struct DigitParams
{
    std::string json_path;    // 模型配置 json 路径
    std::string model_key;    // json 内的模型节点名
    std::string model_folder; // 模型文件夹路径
    int my_color = 2;         // 己方颜色：1=我方蓝 0=我方红 2=离线演示保留双方结果
    float match_dist_ratio = 1.0f; // 匹配阈值：传统装甲板中心与网络装甲板中心的距离
                                   // 不超过 match_dist_ratio * 装甲板宽度 时才认为是同一块
};

// 数字识别器：构造时完成模型加载（失败的时候会抛异常）
class DigitRecognizer
{
public:
    explicit DigitRecognizer(const DigitParams &params);

    // 对整帧图像做一次推理，并把类型匹配给传统方法检测出的装甲板
    void recognize(const cv::Mat &bgr_img, std::vector<Armor> &armors);

    // 仅推理，返回网络原始检测结果
    std::vector<NetArmorResult> detect(const cv::Mat &bgr_img);

    const DigitParams &params() const { return params_; }

private:
    DigitParams params_;
    ArmorModel model_; 
};

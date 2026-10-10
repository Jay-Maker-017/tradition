#pragma once

#include "armor.hpp"
#include "network_deployment_interface.hpp"

#include <string>
#include <vector>

// 装甲板类型（数字）神经网络识别模块
//
// 思路参考 26_Detection_Models-main 开源工程：
//   - 用 NNdeployment 部署库的 ArmorModel 对整帧图像做一次网络推理，得到网络识别出的
//     装甲板（四角点 + 类型/颜色/尺寸）；
//   - 再按中心距与传统方案检测出的装甲板逐一匹配，把类型信息
//     （armor_id / class_name / 置信度）补充到传统 Armor 上。
// 即：传统方案负责定位（角点更精确可控），神经网络负责分类（数字/类型识别）。

// 数字识别参数
struct DigitParams
{
    std::string json_path;    // 模型配置 json 路径（如 configs/detect.json）
    std::string model_key;    // json 内的模型节点名（如 "armor_v8"）
    std::string model_folder; // 模型文件夹路径（json 中 xml 字段相对此文件夹）
    int my_color = 2;         // 己方颜色：1=我方蓝 0=我方红 2=离线演示保留双方结果
    float match_dist_ratio = 1.0f; // 匹配阈值：传统装甲板中心与网络装甲板中心的距离
                                   // 不超过 match_dist_ratio * 装甲板宽度 时才认为是同一块
};

// 数字识别器：构造时完成模型加载（失败抛异常），recognize() 为传统装甲板补充类型
class DigitRecognizer
{
public:
    explicit DigitRecognizer(const DigitParams &params);

    // 对整帧图像做一次推理，并把类型匹配给传统方法检测出的装甲板（原地修改 armors）
    void recognize(const cv::Mat &bgr_img, std::vector<Armor> &armors);

    // 仅推理，返回网络原始检测结果（调试/对比用）
    std::vector<NetArmorResult> detect(const cv::Mat &bgr_img);

    const DigitParams &params() const { return params_; }

private:
    DigitParams params_;
    ArmorModel model_; // 内部基于 YOLOModel 实现（不可复制，可移动）
};

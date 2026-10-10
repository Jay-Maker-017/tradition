#include "armor.hpp"
#include "digit_recognizer.hpp"
#include "lightbar.hpp"
#include "visualize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 默认图片路径
static const std::string kDefaultImagePath =
    "/home/jay/Desktop/auto_aim_homework/tradition/img/test_1.jpg";

// 结果图保存路径
static const std::string kOutputImagePath =
    "/home/jay/Desktop/auto_aim_homework/tradition/lightbar_result.jpg";

// 数字神经网络模型配置（对应 configs/detect.json 中的节点）
static const std::string kConfigPath =
    "/home/jay/Desktop/auto_aim_homework/tradition/configs/detect.json";
static const std::string kModelFolder =
    "/home/jay/Desktop/auto_aim_homework/tradition/models/openvino";
static const std::string kDefaultModelKey = "armor_v8"; // 也可选 armor_v5

//主流程：读图 -> 灯条检测 -> 装甲板配对 -> 数字神经网络类型识别 -> 可视化
int main(int argc, char **argv)
{
    // 读图 (没传参数就用默认绝对路径)
    std::string inPath = (argc >= 2) ? std::string(argv[1]) : kDefaultImagePath;
    // 可选第二个参数：模型节点名（armor_v5 / armor_v8）
    const std::string modelKey = (argc >= 3) ? std::string(argv[2]) : kDefaultModelKey;
    std::cout << "Input image path: " << inPath << std::endl;
    std::cout << "Usage: " << argv[0] << " [input_image] [armor_v5|armor_v8]  (默认使用 " << kDefaultImagePath << ")" << std::endl;

    // 读取图片
    cv::Mat bgr_img = cv::imread(inPath, cv::IMREAD_COLOR);
    if (bgr_img.empty())
    {
        std::cerr << "fail to load image: " << inPath << std::endl;
        return -1;
    }
    std::cout << "Image size: " << bgr_img.cols << "x" << bgr_img.rows << std::endl;

    // 灯条检测
    // LightbarDetector 内部完成：灰度 -> 二值化 -> 找轮廓 -> 几何筛选 -> 颜色判别
    const LightbarDetector detector; // 使用 LightbarParams 默认参数
    const std::vector<Lightbar> lightbars = detector.detect(bgr_img);

    // 打印每根灯条的信息
    for (const auto &bar : lightbars)
    {
        const float width = std::min(bar.rect.size.width, bar.rect.size.height);
        const float height = std::max(bar.rect.size.width, bar.rect.size.height);
        const float ratio = height / width; // 长宽比恒大于 1
        std::cout << "Lightbar color: " << (bar.is_blue ? "Blue" : "Red")
                  << "  center=(" << static_cast<int>(bar.center.x) << ", "
                  << static_cast<int>(bar.center.y) << ")"
                  << "  size=" << std::fixed << std::setprecision(1)
                  << bar.rect.size.width << " x " << bar.rect.size.height // 输出保留一位小数
                  << std::defaultfloat << std::setprecision(6)
                  << "  ratio=" << ratio << std::endl;
    }

    // 装甲板配对
    // 实现思路：对灯条两两打分（颜色,长度,间距,平行/垂直），按得分从优到劣贪心配对，保证每根灯条最多只属于一块装甲板
    const ArmorMatcher matcher;
    std::vector<Armor> armors = matcher.match(lightbars);

    // ==================== 数字神经网络识别装甲板类型 ====================
    // 思路参考 26_Detection_Models-main：用 NNdeployment 的 ArmorModel 对整帧图像做一次
    // 推理，再按中心距把网络识别出的类型（数字/类别）匹配给传统方法检测出的装甲板
    std::cout << "Digit model: " << modelKey << "  (" << kConfigPath << ")" << std::endl;
    try
    {
        const DigitParams digit_params{
            kConfigPath,  // 模型配置 json
            modelKey,     // json 内的模型节点名
            kModelFolder, // 模型文件夹
            2,            // my_color=2：离线演示保留蓝红双方结果
            0.5f,         // 匹配阈值：中心距 <= 0.5 倍装甲板宽度
        };
        DigitRecognizer recognizer(digit_params);
        recognizer.recognize(bgr_img, armors); // 原地为每块装甲板补充类型
    }
    catch (const std::exception &e)
    {
        std::cerr << "[警告] 数字识别模型加载/推理失败: " << e.what() << std::endl;
        std::cerr << "[警告] 已跳过神经网络类型识别，仅输出传统方案结果。" << std::endl;
    }

    std::cout << "Total armors: " << armors.size() << std::endl;
    for (size_t k = 0; k < armors.size(); ++k)
    {
        const Lightbar &left = armors[k].left;   // 左灯条
        const Lightbar &right = armors[k].right; // 右灯条
        const float dx = right.center.x - left.center.x;
        const float dy = right.center.y - left.center.y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        std::cout << "Armor " << (k + 1) << ": " << (left.is_blue ? "Blue" : "Red")
                  << "  left_center=(" << static_cast<int>(left.center.x) << ", "
                  << static_cast<int>(left.center.y) << ")"
                  << "  right_center=(" << static_cast<int>(right.center.x) << ", "
                  << static_cast<int>(right.center.y) << ")"
                  << "  dist=" << std::fixed << std::setprecision(1) << dist
                  << "  score=" << std::setprecision(3) << matcher.pairScore(left, right)
                  << std::defaultfloat << std::setprecision(6) << std::endl;

        // 打印四边形 4 个顶点坐标
        // LT=左上 RT=右上 RB=右下 LB=左下
        const std::vector<cv::Point2f> corners = armorCorners(left, right);
        std::cout << "  Corners: LT=(" << static_cast<int>(corners[0].x) << "," << static_cast<int>(corners[0].y) << ")"
                  << " RT=(" << static_cast<int>(corners[1].x) << "," << static_cast<int>(corners[1].y) << ")"
                  << " RB=(" << static_cast<int>(corners[2].x) << "," << static_cast<int>(corners[2].y) << ")"
                  << " LB=(" << static_cast<int>(corners[3].x) << "," << static_cast<int>(corners[3].y) << ")"
                  << std::endl;

        // 打印数字神经网络识别出的类型（数字）
        if (armors[k].has_type)
        {
            std::cout << "  Type: " << armors[k].class_name << "  id=" << armors[k].armor_id
                      << "  color_id=" << armors[k].color_id
                      << "  score=" << std::fixed << std::setprecision(3) << armors[k].nn_score
                      << std::defaultfloat << std::setprecision(6) << std::endl;
        }
        else
        {
            std::cout << "  Type: 未识别（网络未匹配到该装甲板）" << std::endl;
        }
    }
    std::cout << "Total lightbars: " << lightbars.size() << std::endl;

    // 可视化与输出
    cv::Mat result_img = bgr_img.clone();
    drawLightbars(result_img, lightbars); // 蓝/红灯条旋转框 + 中心点
    drawArmors(result_img, armors);       // 绿色装甲板四边形 + 编号

    // 保存输出结果，方便查看
    cv::imwrite(kOutputImagePath, result_img);
    std::cout << "Result saved to " << kOutputImagePath << std::endl;

    // 显示结果
    if (std::getenv("DISPLAY") != nullptr)
    {
        cv::imshow("4_Lightbars", result_img);
        std::cout << "Press any key on an image window to exit..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    else
    {
        std::cout << "No DISPLAY detected; skip GUI window display. See " << kOutputImagePath << std::endl;
    }

    return 0;
}

#include "armor.hpp"
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

// 默认图片路径（绝对路径）：不带命令行参数时也能直接运行，不受工作目录影响
static const std::string kDefaultImagePath =
    "/home/jay/Desktop/auto_aim_homework/lecture2/img/test_11.png";

// 结果图保存路径（绝对路径）
static const std::string kOutputImagePath =
    "/home/jay/Desktop/auto_aim_homework/lecture2/lightbar_result.jpg";

// ==================== 主流程：读图 → 灯条检测 → 装甲板配对 → 可视化 ====================
int main(int argc, char **argv)
{
    // ---------------- 0. 命令行参数与读图 ----------------
    // 优先使用命令行参数；没传参数就回退到默认绝对路径，避免直接 return 1 退出
    std::string inPath = (argc >= 2) ? std::string(argv[1]) : kDefaultImagePath;
    std::cout << "Input image path: " << inPath << std::endl;
    std::cout << "Usage: " << argv[0] << " [input_image]  (默认使用 " << kDefaultImagePath << ")" << std::endl;

    // 读取图片
    cv::Mat bgr_img = cv::imread(inPath, cv::IMREAD_COLOR);
    if (bgr_img.empty())
    {
        std::cerr << "fail to load image: " << inPath << std::endl;
        return -1;
    }
    std::cout << "Image size: " << bgr_img.cols << "x" << bgr_img.rows << std::endl;

    // ---------------- 1. 灯条检测 ----------------
    // LightbarDetector 内部完成：灰度 → 二值化 → 找轮廓 → 几何筛选 → 颜色判别
    const LightbarDetector detector; // 使用 LightbarParams 默认参数
    const std::vector<Lightbar> lightbars = detector.detect(bgr_img);

    // 打印每根灯条的信息（ratio 用与检测一致的 float 除法复现：长边/短边）
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

    // ---------------- 2. 装甲板配对 ----------------
    // 思路：对灯条两两打分（颜色/长度/间距/平行/垂直判据），
    //       按得分从优到劣贪心配对，保证每根灯条最多只属于一块装甲板。
    const ArmorMatcher matcher; // 使用 ArmorParams 默认参数
    const std::vector<Armor> armors = matcher.match(lightbars);

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

        // 打印四边形 4 个顶点坐标（与图上绿色四边形一一对应）：
        // 顺序为 左上 右上 右下 左下，即 LT=左上 RT=右上 RB=右下 LB=左下
        const std::vector<cv::Point2f> corners = armorCorners(left, right);
        std::cout << "  Corners: LT=(" << static_cast<int>(corners[0].x) << "," << static_cast<int>(corners[0].y) << ")"
                  << " RT=(" << static_cast<int>(corners[1].x) << "," << static_cast<int>(corners[1].y) << ")"
                  << " RB=(" << static_cast<int>(corners[2].x) << "," << static_cast<int>(corners[2].y) << ")"
                  << " LB=(" << static_cast<int>(corners[3].x) << "," << static_cast<int>(corners[3].y) << ")"
                  << std::endl;
    }
    std::cout << "Total lightbars: " << lightbars.size() << std::endl;

    // ---------------- 3. 可视化与输出 ----------------
    // 所有绘制都画在副本上，不污染 bgr_img
    cv::Mat result_img = bgr_img.clone();
    drawLightbars(result_img, lightbars); // 蓝/红灯条旋转框 + 中心点
    drawArmors(result_img, armors);       // 绿色装甲板四边形 + 编号

    // 存盘，方便无图形环境时查看结果
    cv::imwrite(kOutputImagePath, result_img);
    std::cout << "Result saved to " << kOutputImagePath << std::endl;

    // Show result
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

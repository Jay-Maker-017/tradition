#include "armor.hpp"
#include "digit_recognizer.hpp"
#include "lightbar.hpp"
#include "visualize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 默认视频路径；不带参数时处理 tradition/video 中的测试视频。
static const std::string kDefaultVideoPath =
    "/home/jay/Desktop/auto_aim_homework/tradition/video/video_combat.avi";

// 默认图片路径（用于显示用法示例）
static const std::string kDefaultImagePath =
    "/home/jay/Desktop/auto_aim_homework/tradition/img/test_1.jpg";

// 结果图片/视频保存路径
static const std::string kOutputImagePath =
    "/home/jay/Desktop/auto_aim_homework/tradition/lightbar_result.jpg";
static const std::string kOutputVideoPath =
    "/home/jay/Desktop/auto_aim_homework/tradition/lightbar_result.avi";

// 数字神经网络模型配置（对应 configs/detect.json 中的节点）
static const std::string kConfigPath =
    "/home/jay/Desktop/auto_aim_homework/tradition/configs/detect.json";
static const std::string kModelFolder =
    "/home/jay/Desktop/auto_aim_homework/tradition/models/openvino";
static const std::string kDefaultModelKey = "armor_v8"; // 也可选 armor_v5

namespace
{
void processFrame(const cv::Mat &bgr_img, const LightbarDetector &detector,
                  const ArmorMatcher &matcher, DigitRecognizer *recognizer,
                  bool &digit_recognition_enabled,
                  std::vector<Lightbar> &lightbars, std::vector<Armor> &armors)
{
    lightbars = detector.detect(bgr_img);
    armors = matcher.match(lightbars);
    if (recognizer != nullptr && digit_recognition_enabled)
    {
        try
        {
            recognizer->recognize(bgr_img, armors);
        }
        catch (const std::exception &e)
        {
            digit_recognition_enabled = false;
            std::cerr << "[警告] 数字识别推理失败: " << e.what()
                      << "；后续帧仅运行传统检测。" << std::endl;
        }
    }
}

void printImageDetails(const std::vector<Lightbar> &lightbars,
                       const std::vector<Armor> &armors,
                       const ArmorMatcher &matcher)
{
    for (const auto &bar : lightbars)
    {
        const float width = std::min(bar.rect.size.width, bar.rect.size.height);
        const float height = std::max(bar.rect.size.width, bar.rect.size.height);
        const float ratio = width > 0.f ? height / width : 0.f;
        std::cout << "Lightbar color: " << (bar.is_blue ? "Blue" : "Red")
                  << "  center=(" << static_cast<int>(bar.center.x) << ", "
                  << static_cast<int>(bar.center.y) << ")"
                  << "  size=" << std::fixed << std::setprecision(1)
                  << bar.rect.size.width << " x " << bar.rect.size.height
                  << std::defaultfloat << std::setprecision(6)
                  << "  ratio=" << ratio << std::endl;
    }

    std::cout << "Total armors: " << armors.size() << std::endl;
    for (size_t k = 0; k < armors.size(); ++k)
    {
        const Lightbar &left = armors[k].left;
        const Lightbar &right = armors[k].right;
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

        const std::vector<cv::Point2f> corners = armorCorners(left, right);
        std::cout << "  Corners: LT=(" << static_cast<int>(corners[0].x) << "," << static_cast<int>(corners[0].y) << ")"
                  << " RT=(" << static_cast<int>(corners[1].x) << "," << static_cast<int>(corners[1].y) << ")"
                  << " RB=(" << static_cast<int>(corners[2].x) << "," << static_cast<int>(corners[2].y) << ")"
                  << " LB=(" << static_cast<int>(corners[3].x) << "," << static_cast<int>(corners[3].y) << ")"
                  << std::endl;

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
}

void printVideoSummary(int frame_index, int total_frames,
                       const std::vector<Lightbar> &lightbars,
                       const std::vector<Armor> &armors)
{
    std::cout << "[帧 " << frame_index;
    if (total_frames > 0)
        std::cout << "/" << total_frames;
    std::cout << "] lightbars=" << lightbars.size() << " armors=" << armors.size();
    for (const Armor &armor : armors)
    {
        std::cout << " [" << (armor.left.is_blue ? "Blue" : "Red") << "-";
        if (armor.has_type)
            std::cout << armor.class_name << " " << std::fixed << std::setprecision(2) << armor.nn_score;
        else
            std::cout << "未识别";
        std::cout << "]";
    }
    std::cout << std::defaultfloat << std::setprecision(6) << std::endl;
}

cv::Mat makeAnnotatedFrame(const cv::Mat &frame,
                           const std::vector<Lightbar> &lightbars,
                           const std::vector<Armor> &armors)
{
    cv::Mat result = frame.clone();
    drawLightbars(result, lightbars);
    drawArmors(result, armors);
    return result;
}

cv::Mat resizeForDisplay(const cv::Mat &image)
{
    constexpr int min_display_width = 460;
    constexpr int max_display_width = 1280;
    cv::Mat display = image;
    if (display.cols < min_display_width || display.cols > max_display_width)
    {
        const int target_width = std::clamp(display.cols, min_display_width, max_display_width);
        const double scale = static_cast<double>(target_width) / display.cols;
        cv::resize(image, display, cv::Size(), scale, scale);
    }
    return display;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc > 3)
    {
        std::cerr << "用法: " << argv[0] << " [图片或视频路径] [armor_v5|armor_v8]" << std::endl;
        return 2;
    }

    const std::string input_path = argc >= 2 ? argv[1] : kDefaultVideoPath;
    const std::string model_key = argc >= 3 ? argv[2] : kDefaultModelKey;
    if (model_key != "armor_v5" && model_key != "armor_v8")
    {
        std::cerr << "未知模型节点: " << model_key << "（可选 armor_v5 或 armor_v8）" << std::endl;
        return 2;
    }
    std::cout << "用法: " << argv[0] << " [图片或视频路径] [armor_v5|armor_v8]" << std::endl;
    std::cout << "默认输入: " << kDefaultVideoPath << std::endl;

    // 不以帧数属性判断输入类型：部分 PNG/RGBA 图片的 FRAME_COUNT 会返回无效值。
    // 先用 imread 试读；读出非空 BGR 图像就是单张图片，否则再尝试打开视频。
    cv::Mat still_image = cv::imread(input_path, cv::IMREAD_COLOR);
    const bool is_single_image = !still_image.empty();
    cv::VideoCapture capture;
    double source_fps = 0.0;
    int total_frames = 0;
    cv::Size frame_size;
    if (is_single_image)
    {
        frame_size = still_image.size();
        std::cout << "输入源: 图片 " << input_path << " (" << frame_size.width << "x"
                  << frame_size.height << ", " << still_image.channels() << " 通道)" << std::endl;
    }
    else
    {
        capture.open(input_path);
        if (!capture.isOpened())
        {
            std::cerr << "无法将输入作为图片或视频打开: " << input_path << std::endl;
            return 1;
        }
        source_fps = capture.get(cv::CAP_PROP_FPS);
        const double reported_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (reported_frames > 0.0 && reported_frames <= std::numeric_limits<int>::max())
            total_frames = static_cast<int>(reported_frames);
        frame_size = cv::Size(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
                              static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
        std::cout << "输入源: 视频 " << input_path << " (" << frame_size.width << "x"
                  << frame_size.height << ", " << source_fps << " FPS, "
                  << total_frames << " 帧)" << std::endl;
    }

    const bool show_gui = std::getenv("DISPLAY") != nullptr;
    const LightbarDetector detector;
    const ArmorMatcher matcher;
    std::unique_ptr<DigitRecognizer> recognizer;
    std::cout << "Digit model: " << model_key << " (" << kConfigPath << ")" << std::endl;
    try
    {
        const DigitParams digit_params{kConfigPath, model_key, kModelFolder, 2, 0.5f};
        recognizer = std::make_unique<DigitRecognizer>(digit_params);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[警告] 数字识别模型加载失败: " << e.what() << std::endl;
        std::cerr << "[警告] 将仅输出传统方案结果。" << std::endl;
    }

    std::vector<Lightbar> lightbars;
    std::vector<Armor> armors;
    bool digit_recognition_enabled = recognizer != nullptr;
    if (is_single_image)
    {
        processFrame(still_image, detector, matcher, recognizer.get(),
                     digit_recognition_enabled, lightbars, armors);
        printImageDetails(lightbars, armors, matcher);
        const cv::Mat result = makeAnnotatedFrame(still_image, lightbars, armors);
        if (!cv::imwrite(kOutputImagePath, result))
        {
            std::cerr << "无法保存结果图片: " << kOutputImagePath << std::endl;
            return 1;
        }
        std::cout << "结果图片已保存: " << kOutputImagePath << std::endl;
        if (show_gui)
        {
            cv::imshow("Tradition Armor Detection", result);
            std::cout << "按任意键关闭图片窗口。" << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        else
        {
            std::cout << "未检测到 DISPLAY，跳过窗口显示。" << std::endl;
        }
        return 0;
    }

    const double output_fps = source_fps > 0.0 && std::isfinite(source_fps) ? source_fps : 30.0;
    cv::VideoWriter writer;
    writer.open(kOutputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                output_fps, frame_size, true);
    if (!writer.isOpened())
    {
        std::cerr << "[警告] 无法创建 AVI 结果视频: " << kOutputVideoPath
                  << "；将继续处理但不保存视频。" << std::endl;
    }

    const double ticks_per_second = cv::getTickFrequency();
    const double frame_period_ms = 1000.0 / output_fps;
    const int64_t start_ticks = cv::getTickCount();
    int processed_frames = 0;
    cv::Mat frame;
    while (capture.read(frame))
    {
        const int64_t frame_start_ticks = cv::getTickCount();
        processFrame(frame, detector, matcher, recognizer.get(),
                 digit_recognition_enabled, lightbars, armors);
        ++processed_frames;
        printVideoSummary(processed_frames, total_frames, lightbars, armors);

        const cv::Mat result = makeAnnotatedFrame(frame, lightbars, armors);
        if (writer.isOpened())
            writer.write(result);

        if (show_gui)
        {
            cv::imshow("Tradition Armor Detection", resizeForDisplay(result));
            const double elapsed_ms = (cv::getTickCount() - frame_start_ticks) * 1000.0 / ticks_per_second;
            const int wait_ms = std::max(1, static_cast<int>(std::ceil(frame_period_ms - elapsed_ms)));
            const int key = cv::waitKey(wait_ms);
            if (key == 'q' || key == 'Q' || key == 27)
            {
                std::cout << "收到退出按键，提前结束视频处理。" << std::endl;
                break;
            }
        }
    }
    const bool video_saved = writer.isOpened() && processed_frames > 0;
    writer.release();
    if (show_gui)
        cv::destroyAllWindows();

    const double total_seconds = (cv::getTickCount() - start_ticks) / ticks_per_second;
    const double average_ms = processed_frames > 0 ? total_seconds * 1000.0 / processed_frames : 0.0;
    const double average_fps = total_seconds > 0.0 ? processed_frames / total_seconds : 0.0;
    std::cout << "视频处理结束：共处理 " << processed_frames << " 帧，耗时 "
              << std::fixed << std::setprecision(2) << total_seconds << " 秒，平均 "
              << average_ms << " ms/帧（" << average_fps << " FPS）。"
              << std::defaultfloat << std::setprecision(6) << std::endl;
    if (video_saved)
        std::cout << "结果视频已保存: " << kOutputVideoPath << std::endl;
    else if (processed_frames > 0)
        std::cout << "结果视频未保存（VideoWriter 初始化失败）。" << std::endl;

    return 0;
}

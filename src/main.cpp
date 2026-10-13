#include "pipeline.hpp"
#include "pnp_solver.hpp"
#include "report.hpp"
#include "visualize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <string>

namespace
{
// 默认输入与结果输出路径
// (之前一直出现找不到文件的错误，很烦，所以干脆用绝对路径了)
const std::string kDefaultVideoPath =
    "/home/jay/Desktop/auto_aim_homework/tradition/video/video_combat.avi";
[[maybe_unused]] const std::string kDefaultImagePath =
    "/home/jay/Desktop/auto_aim_homework/tradition/img/test_1.jpg";
const std::string kOutputImagePath =
    "/home/jay/Desktop/auto_aim_homework/tradition/lightbar_result.jpg";
const std::string kOutputVideoPath =
    "/home/jay/Desktop/auto_aim_homework/tradition/lightbar_result.avi";

// 数字神经网络模型配置
const std::string kConfigPath =
    "/home/jay/Desktop/auto_aim_homework/tradition/configs/detect.json";
const std::string kCameraConfigPath = "configs/camera.json";
const std::string kModelFolder =
    "/home/jay/Desktop/auto_aim_homework/tradition/models/openvino";
const std::string kDefaultModelKey = "armor_v8";

struct AppOptions
{
    std::string input_path;
    std::string model_key;
};

struct VideoInfo
{
    double source_fps = 0.0;
    int total_frames = 0;
    cv::Size frame_size;
};

bool parseArgs(int argc, char **argv, AppOptions &options)
{
    if (argc > 3)
    {
        //std::cerr << "用法: " << argv[0] << " [图片或视频路径] [armor_v5|armor_v8]" << std::endl;
        return false;
    }

    options.input_path = argc >= 2 ? argv[1] : kDefaultVideoPath;
    options.model_key = argc >= 3 ? argv[2] : kDefaultModelKey;
    if (options.model_key != "armor_v5" && options.model_key != "armor_v8")
    {
        //std::cerr << "未知模型节点: " << options.model_key << "（可选 armor_v5 或 armor_v8）" << std::endl;
        return false;
    }

    //std::cout << "用法: " << argv[0] << " [图片或视频路径] [armor_v5|armor_v8]" << std::endl;
    //std::cout << "默认输入: " << kDefaultVideoPath << std::endl;
    return true;
}

void printModelLoadWarning(const DetectionPipeline &pipeline)
{
    if (!pipeline.recognitionError().empty())
    {
        std::cerr << "[警告] 数字识别模型加载失败: " << pipeline.recognitionError() << std::endl;
        std::cerr << "[警告] 将仅输出传统方案结果。" << std::endl;
    }
}

int runImageMode(const cv::Mat &still_image, DetectionPipeline &pipeline,
                 const PnPSolver &pnp_solver, bool show_gui)
{
    DetectionResult detection = pipeline.processFrame(still_image);
    printImageDetails(detection.lightbars, detection.armors, pipeline.matcher());

    const std::vector<PnPResult> pnp_results = pnp_solver.solveAll(detection.armors);
    printPnPResults(detection.armors, pnp_results);

    cv::Mat result = makeAnnotatedFrame(still_image, detection.armors);
    drawPnPResults(result, pnp_results, pnp_solver.cameraParams());
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

int runVideoMode(cv::VideoCapture &capture, const VideoInfo &video_info,
                 DetectionPipeline &pipeline, bool show_gui)
{
    const double output_fps = video_info.source_fps > 0.0 && std::isfinite(video_info.source_fps)
                                  ? video_info.source_fps
                                  : 30.0;
    cv::VideoWriter writer;
    writer.open(kOutputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                output_fps, video_info.frame_size, true);
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
        DetectionResult detection = pipeline.processFrame(frame);
        ++processed_frames;
        printVideoSummary(processed_frames, video_info.total_frames,
                          detection.lightbars, detection.armors);

        const cv::Mat result = makeAnnotatedFrame(frame, detection.armors);
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
} // namespace

int main(int argc, char **argv)
{
    AppOptions options;
    if (!parseArgs(argc, argv, options))
        return 2;

    // 先尝试读取单张图片，再尝试打开视频；不依赖可能无效的图片帧数属性。
    cv::Mat still_image = cv::imread(options.input_path, cv::IMREAD_COLOR);
    const bool is_single_image = !still_image.empty();
    cv::VideoCapture capture;
    VideoInfo video_info;
    if (is_single_image)
    {
        const cv::Size frame_size = still_image.size();
        std::cout << "输入源: 图片 " << options.input_path << " (" << frame_size.width << "x"
                  << frame_size.height << ", " << still_image.channels() << " 通道)" << std::endl;
    }
    else
    {
        capture.open(options.input_path);
        if (!capture.isOpened())
        {
            std::cerr << "无法将输入作为图片或视频打开: " << options.input_path << std::endl;
            return 1;
        }
        video_info.source_fps = capture.get(cv::CAP_PROP_FPS);
        const double reported_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (reported_frames > 0.0 && reported_frames <= std::numeric_limits<int>::max())
            video_info.total_frames = static_cast<int>(reported_frames);
        video_info.frame_size = cv::Size(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
                                         static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
        std::cout << "输入源: 视频 " << options.input_path << " (" << video_info.frame_size.width << "x"
                  << video_info.frame_size.height << ", " << video_info.source_fps << " FPS, "
                  << video_info.total_frames << " 帧)" << std::endl;
    }

    const bool show_gui = std::getenv("DISPLAY") != nullptr;
    std::cout << "Digit model: " << options.model_key << " (" << kConfigPath << ")" << std::endl;
    const DigitParams digit_params{kConfigPath, options.model_key, kModelFolder, 2, 0.5f};
    DetectionPipeline pipeline(digit_params);
    printModelLoadWarning(pipeline);

    if (is_single_image)
    {
        const CameraParams calibrated_camera = loadCameraParams(kCameraConfigPath);
        const CameraParams camera = adaptCameraParams(calibrated_camera, still_image.size());
        const PnPSolver pnp_solver(camera);
        return runImageMode(still_image, pipeline, pnp_solver, show_gui);
    }
    return runVideoMode(capture, video_info, pipeline, show_gui);
}

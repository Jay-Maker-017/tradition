#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ==================== 装甲板配对：数据结构与参数 ====================

// 单根灯条的描述（从已经过几何筛选的 minAreaRect 提取而来）
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

// ---- 配对参数（集中在此处，便于按不同图片微调）----
static const float kMaxLenDiffRatio = 0.35f; // 长度差/较长者 的允许上限（两灯条长度需接近）
static const float kMinDistRatio = 0.8f;     // 中心距/平均长度 的下限（太近不算一块板）
static const float kMaxDistRatio = 4.5f;     // 中心距/平均长度 的上限（太远不算一块板）
static const float kIdealDistRatio = 2.6f;   // 理想间距比（按实测测试图 ≈2.6~2.8 取值，用于打分，越接近越优）
static const float kMaxParallelErr = 0.2f;   // 平行误差上限（对应两长轴夹角约 11.5°）
static const float kMaxPerpErr = 0.35f;      // 垂直误差上限（对应中心连线与长轴夹角约 20°）
// 得分项权重（默认均为 1.0，按需调整可改变各项判据的话语权）
static const float wLenDiff = 1.0f;          // 长度差
static const float wParallel = 1.0f;         // 平行误差
static const float wPerp = 1.0f;             // 垂直误差
static const float wDist = 1.0f;             // 间距偏离理想值

// 由 minAreaRect 结果构造 Lightbar：
// 用四点中"最长的那条边"确定长轴方向（不依赖 RotatedRect::angle，避免 OpenCV 版本差异）
static Lightbar makeLightbar(const cv::RotatedRect &rect, bool is_blue, int id)
{
    Lightbar bar;
    bar.rect = rect;
    bar.center = rect.center;
    bar.length = std::max(rect.size.width, rect.size.height);
    bar.width = std::min(rect.size.width, rect.size.height);
    bar.is_blue = is_blue;
    bar.id = id;

    // 取出四个角点，找出最长边（即长轴方向）
    cv::Point2f pts[4];
    rect.points(pts);
    cv::Point2f best(0.f, 0.f);
    float best_len = 0.f;
    for (int i = 0; i < 4; ++i)
    {
        cv::Point2f d = pts[(i + 1) % 4] - pts[i];
        float len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len > best_len)
        {
            best_len = len;
            best = d;
        }
    }
    // 归一化成单位方向向量；方向正负无所谓（后续判据都取绝对值）
    if (best_len > 1e-6f)
        bar.axis = best / best_len;
    else
        bar.axis = cv::Point2f(0.f, 1.f);
    return bar;
}

// 计算两根灯条构成装甲板的得分：返回 -1 表示不满足硬判据；否则得分越小越优
static float pairScore(const Lightbar &a, const Lightbar &b)
{
    // 判据 1：颜色必须一致（两蓝或两红才能组成一块板）
    if (a.is_blue != b.is_blue)
        return -1.f;

    // 判据 2：两灯条长度要接近（长度差 / 较长者）
    const float max_len = std::max(a.length, b.length);
    if (max_len <= 1e-6f)
        return -1.f;
    const float len_diff = std::fabs(a.length - b.length) / max_len;
    if (len_diff > kMaxLenDiffRatio)
        return -1.f;

    // 判据 3：中心距与平均长度之比要落在合理区间（太近/太远都不是一块板）
    const cv::Point2f dc = b.center - a.center;
    const float dist = std::sqrt(dc.x * dc.x + dc.y * dc.y);
    const float avg_len = 0.5f * (a.length + b.length);
    if (avg_len <= 1e-6f)
        return -1.f;
    const float dist_ratio = dist / avg_len;
    if (dist_ratio < kMinDistRatio || dist_ratio > kMaxDistRatio)
        return -1.f;

    // 判据 4：两长轴方向要接近平行
    //   cross = |sin(夹角)|，0 表示完全平行，1 表示完全垂直（方向正负不影响，故取绝对值）
    const float cross = std::fabs(a.axis.x * b.axis.y - a.axis.y * b.axis.x);
    if (cross > kMaxParallelErr)
        return -1.f;

    // 判据 5：两灯条中心连线要近似垂直于长轴（即"并排"而不是"首尾相接"）
    const cv::Point2f dir = dc / std::max(dist, 1e-6f); // 连线方向单位向量
    const float dot = std::fabs(dir.x * a.axis.x + dir.y * a.axis.y);
    if (dot > kMaxPerpErr)
        return -1.f;

    // 全部硬判据通过 → 计算加权得分
    // 各项误差都归一化到相近量级（0~1），加权求和，越小越优
    const float dist_err = std::fabs(dist_ratio - kIdealDistRatio) / kIdealDistRatio;
    return wLenDiff * len_diff + wParallel * cross + wPerp * dot + wDist * dist_err;
}

// 贪心一对一配对：生成所有候选对 → 按得分从优到劣排序 → 每根灯条最多只用一次
// 返回的 pair 中 first 为左灯条、second 为右灯条（按 x 坐标摆放）
static std::vector<std::pair<Lightbar, Lightbar>> matchLightbars(const std::vector<Lightbar> &bars)
{
    // 候选对：{得分, 灯条 i, 灯条 j}
    struct Candidate
    {
        float score;
        int i;
        int j;
    };
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < bars.size(); ++i)
    {
        for (size_t j = i + 1; j < bars.size(); ++j)
        {
            const float s = pairScore(bars[i], bars[j]);
            if (s >= 0.f)
                candidates.push_back({s, static_cast<int>(i), static_cast<int>(j)});
        }
    }

    // 得分升序：越优（越小）的候选对越先被接受
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) { return a.score < b.score; });

    std::vector<bool> used(bars.size(), false);
    std::vector<std::pair<Lightbar, Lightbar>> pairs;
    for (const auto &c : candidates)
    {
        if (used[c.i] || used[c.j])
            continue; // 任意一根已被占用则跳过，保证一对一
        used[c.i] = true;
        used[c.j] = true;
        // 统一按 x 坐标摆放：左灯条在前、右灯条在后
        if (bars[c.i].center.x <= bars[c.j].center.x)
            pairs.push_back({bars[c.i], bars[c.j]});
        else
            pairs.push_back({bars[c.j], bars[c.i]});
    }

    // 最后按装甲板中心 x 坐标从左到右排序，保证控制台打印与图上 "Armor N" 编号一致
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<Lightbar, Lightbar> &a, const std::pair<Lightbar, Lightbar> &b) {
                  return 0.5f * (a.first.center.x + a.second.center.x) <
                         0.5f * (b.first.center.x + b.second.center.x);
              });
    return pairs;
}

// 提取单根灯条长轴方向的两个端点（端面中点）：
// 把 4 个角点投影到"两灯条中心连线的垂线"perp 上，
// 投影最大的两个角点属于同一端（up），最小的两个属于另一端（down）；
// 每端取两角点的中点作为端点 → 四边形宽边穿过灯条中心线，左右对称。
// 注意：不直接用 Lightbar::axis 定端（其正负取决于角点遍历顺序、并不固定），
//       用 perp 投影判定才能保证两根灯条的上下端严格一一对应，避免四边形交叉成蝴蝶结。
static void barEndpoints(const Lightbar &bar, const cv::Point2f &perp,
                         cv::Point2f &up, cv::Point2f &down)
{
    cv::Point2f pts[4];
    bar.rect.points(pts);
    // 按在 perp 上的投影值升序排列（投影值 = 角点与 perp 的点积）
    std::sort(pts, pts + 4, [&perp](const cv::Point2f &a, const cv::Point2f &b) {
        return (a.x * perp.x + a.y * perp.y) < (b.x * perp.x + b.y * perp.y);
    });
    down = 0.5f * (pts[0] + pts[1]); // 投影最小的两个角点 → 下端中点
    up = 0.5f * (pts[2] + pts[3]);   // 投影最大的两个角点 → 上端中点
}

// 计算一块装甲板的 4 个角点（四边形顶点）：
// 每根灯条取长轴方向的两个端点，共 4 个点；
// 返回顺序固定为 [左上, 右上, 右下, 左下]（左=左灯条、右=右灯条，pairs 中已按 x 排好），
// 供控制台打印与绘图共用，保证"打印的坐标"就是"图上画的顶点"。
static std::vector<cv::Point2f> armorCorners(const Lightbar &left, const Lightbar &right)
{
    // 两灯条中心连线的方向向量及其垂线
    const cv::Point2f dc = right.center - left.center;
    const float dist = std::max(std::sqrt(dc.x * dc.x + dc.y * dc.y), 1e-6f); // 防除零
    const cv::Point2f dir = dc / dist; // 连线单位向量（左灯条 → 右灯条）
    cv::Point2f perp(-dir.y, dir.x);   // 与连线垂直的单位向量
    if (perp.y > 0.f)
        perp = -perp; // 让 perp 指向图像上方（y 减小方向），保证"上端"确为视觉上的上方

    cv::Point2f lt, lb, rt, rb;
    barEndpoints(left, perp, lt, lb);  // 左灯条的上、下端
    barEndpoints(right, perp, rt, rb); // 右灯条的上、下端

    return {lt, rt, rb, lb}; // 左上 → 右上 → 右下 → 左下，闭合即四边形
}

// 在图上绘制已配对的装甲板：把两灯条的四个端点连成绿色四边形（贴合装甲板真实轮廓）
static void drawArmors(cv::Mat &img,
                       const std::vector<std::pair<Lightbar, Lightbar>> &pairs)
{
    if (pairs.empty())
    {
        // 没有配对成功的装甲板时给个文字提示，便于确认这一环节确实跑了
        cv::putText(img, "No armor detected", cv::Point(15, img.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        return;
    }

    // 绿色：装甲板轮廓（与单灯条的红/蓝框区分开）
    const cv::Scalar armor_color(0, 255, 0);
    for (size_t k = 0; k < pairs.size(); ++k)
    {
        // 四个顶点：[左上, 右上, 右下, 左下]（与 main 里打印的坐标完全一致）
        const std::vector<cv::Point2f> corners = armorCorners(pairs[k].first, pairs[k].second);

        // polylines 要求整数点集（CV_32S），先四舍五入转换
        std::vector<cv::Point> int_corners;
        int_corners.reserve(corners.size());
        for (const auto &p : corners)
            int_corners.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));

        // 用 polylines 一次画闭合四边形（isClosed=true 自动首尾相连）
        cv::polylines(img, int_corners, /*isClosed=*/true, armor_color, 2, cv::LINE_AA);

        // 板中心 = 四个顶点的形心（均值），画实心点 + 编号文字
        cv::Point2f center(0.f, 0.f);
        float top_y = corners[0].y; // 四边形最高点 y，用于把文字放在框上方
        for (const auto &p : corners)
        {
            center += p;
            top_y = std::min(top_y, p.y);
        }
        center *= 0.25f;
        cv::circle(img, center, 4, armor_color, cv::FILLED, cv::LINE_AA);

        const std::string label = "Armor " + std::to_string(k + 1);
        int base_line = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base_line);
        // 文字放在四边形上方水平居中，并保证不超出图像上边界
        cv::Point org(static_cast<int>(center.x - text_size.width / 2.f),
                      std::max(text_size.height + 4, static_cast<int>(top_y) - 8));
        cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.6, armor_color, 2, cv::LINE_AA);
    }
}

// =====================================================================

// 默认图片路径（绝对路径）：不带命令行参数时也能直接运行，不受工作目录影响
static const std::string kDefaultImagePath =
    "/home/jay/Desktop/auto_aim_homework/lecture2/img/test_11.png";


int main(int argc, char **argv)
{
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

    // 结果图画在副本上，避免画框污染 bgr_img 影响后面的颜色统计
    cv::Mat result_img = bgr_img.clone();
    // 转灰度
    cv::Mat gray_img;
    cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
    //cv::imshow("1_gray_img", gray_img);
    // 二值化
    double threshold_ = 200;
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
    //cv::imshow("2_binary_img", binary_img);
    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat contour_img = bgr_img.clone();
    cv::drawContours(contour_img, contours, -1, cv::Scalar(0, 0, 255), 6);
    //cv::imshow("3_contours", contour_img);
    // 参数
    double min_lightbar_ratio_ = 3.0;  // 最小长宽比
    double max_lightbar_ratio_ = 9.0;  // 最大长宽比
    double min_lightbar_length_ = 5.0; // 最小长度

    std::vector<cv::Mat> lightbar_patches; // 抠出来的灯条小图，用于并排展示
    int lightbar_count = 0;                // 检出的灯条总数

    // 新增：收集通过筛选的灯条几何信息，供循环结束后两两配对成装甲板
    std::vector<Lightbar> lightbars;

    // 遍历轮廓
    for (const auto &contour : contours)
    {
        if (contour.size() < 18)
            continue; // 将太小的轮廓忽略
        auto rect = cv::minAreaRect(contour);

        // 计算长宽
        float width = std::min(rect.size.width, rect.size.height);
        float height = std::max(rect.size.width, rect.size.height);
        if (width <= 0.f)
            continue; // 防止除零
        float ratio = height / width; // 长宽比恒大于1

        // 几何筛选
        if (ratio > min_lightbar_ratio_ && ratio < max_lightbar_ratio_ &&
            height > min_lightbar_length_)
        {
            // 判断颜色：用灯条区域内所有像素的均值比较蓝/红通道，
            // 比只看轮廓边界点更稳定
            cv::Mat mask = cv::Mat::zeros(bgr_img.size(), CV_8UC1);
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0,
                             cv::Scalar(255), cv::FILLED);
            cv::Scalar mean = cv::mean(bgr_img, mask);
            double blue_mean = mean[0]; // B
            double red_mean = mean[2];  // R

            bool is_blue = (blue_mean > red_mean);

            // 蓝灯条画蓝框，红灯条画红框
            cv::Scalar color = is_blue ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255);

            // 绘制旋转矩形（画在 result_img 上，不污染 bgr_img）
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++)
            {
                cv::line(result_img, vertices[i], vertices[(i + 1) % 4], color, 3, cv::LINE_AA);
            }
            cv::circle(result_img, rect.center, 3, cv::Scalar(0, 255, 0), -1); // 中心点

            // 抠出灯条小图（统一摆成横向，方便并排）（后面的显示暂时关掉了）
            cv::Size patch_size(std::max(1.f, rect.size.width), std::max(1.f, rect.size.height));
            cv::Mat patch;
            cv::getRectSubPix(bgr_img, patch_size, rect.center, patch);
            if (!patch.empty())
            {
                if (patch.rows > patch.cols)
                    cv::rotate(patch, patch, cv::ROTATE_90_CLOCKWISE);
                lightbar_patches.push_back(patch);
            }

            // 新增：把这根合格灯条的几何信息存起来（id 即当前检出序号），供配对使用
            lightbars.push_back(makeLightbar(rect, is_blue, lightbar_count));

            ++lightbar_count;
            std::cout << "Lightbar color: " << (is_blue ? "Blue" : "Red")
                      << "  center=(" << static_cast<int>(rect.center.x) << ", "
                      << static_cast<int>(rect.center.y) << ")"
                      << "  size=" << std::fixed << std::setprecision(1)
                      << rect.size.width << " x " << rect.size.height  //输出保留一位小数
                      << std::defaultfloat << std::setprecision(6)
                      << "  ratio=" << ratio << std::endl;
        }
    }

    // ======================= 装甲板配对（两根灯条组成一块板） =======================
    // 思路：对收集到的灯条两两打分（颜色/长度/间距/平行/垂直判据），
    //       按得分从优到劣贪心配对，保证每根灯条最多只属于一块装甲板。
    const auto armor_pairs = matchLightbars(lightbars);
    std::cout << "Total armors: " << armor_pairs.size() << std::endl;
    for (size_t k = 0; k < armor_pairs.size(); ++k)
    {
        const Lightbar &left = armor_pairs[k].first;   // 左灯条
        const Lightbar &right = armor_pairs[k].second; // 右灯条
        const float dx = right.center.x - left.center.x;
        const float dy = right.center.y - left.center.y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        std::cout << "Armor " << (k + 1) << ": " << (left.is_blue ? "Blue" : "Red")
                  << "  left_center=(" << static_cast<int>(left.center.x) << ", "
                  << static_cast<int>(left.center.y) << ")"
                  << "  right_center=(" << static_cast<int>(right.center.x) << ", "
                  << static_cast<int>(right.center.y) << ")"
                  << "  dist=" << std::fixed << std::setprecision(1) << dist
                  << "  score=" << std::setprecision(3) << pairScore(left, right)
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
    // 把装甲板框画到 result_img 上，随下面的 imwrite / imshow 一起输出
    drawArmors(result_img, armor_pairs);
    // ==============================================================================

    std::cout << "Total lightbars: " << lightbar_count << std::endl;

    // 把所有灯条小图并排拼成一张总览图
    cv::Mat gallery;
    if (!lightbar_patches.empty())
    {
        int max_h = 0;
        for (const auto &p : lightbar_patches)
            max_h = std::max(max_h, p.rows);

        const int gap = 6;
        const int top = 24;
        int total_w = gap;
        for (const auto &p : lightbar_patches)
            total_w += p.cols + gap;

        gallery = cv::Mat::zeros(max_h + top, total_w, CV_8UC3);
        int x = gap;
        for (const auto &p : lightbar_patches)
        {
            p.copyTo(gallery(cv::Rect(x, top, p.cols, p.rows)));
            x += p.cols + gap;
        }
        cv::putText(gallery, "detected lightbars", cv::Point(8, 17),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    // 存盘，方便无图形环境时查看结果
    const std::string outPath =
        "/home/jay/Desktop/auto_aim_homework/lecture2/lightbar_result.jpg";
    cv::imwrite(outPath, result_img);
    std::cout << "Result saved to " << outPath << std::endl;

    // Show result
    if (std::getenv("DISPLAY") != nullptr)
    {
        cv::imshow("4_Lightbars", result_img);
        if (!gallery.empty()){
            // cv::imshow("5_lightbar_gallery", gallery);
            std::cout << "Press any key on an image window to exit..." << std::endl;}
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    else
    {
        std::cout << "No DISPLAY detected; skip GUI window display. See " << outPath << std::endl;
    }

    return 0;
}

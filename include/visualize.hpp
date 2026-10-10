#pragma once

#include "armor.hpp"
#include "lightbar.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

// 可视化模块

// 绘制单根灯条：蓝灯条画蓝框、红灯条画红框，中心画绿色实心点
// draw_rects 为 false 时不画红蓝矩形框，只画中心点
void drawLightbars(cv::Mat &img, const std::vector<Lightbar> &lightbars,
                   bool draw_rects = true);

// 绘制已配对的装甲板
void drawArmors(cv::Mat &img, const std::vector<Armor> &armors);

// 生成带灯条和装甲板标注的画面。
cv::Mat makeAnnotatedFrame(const cv::Mat &frame,
                           const std::vector<Armor> &armors);

// 将画面宽度缩放到适合窗口显示的范围。
cv::Mat resizeForDisplay(const cv::Mat &image);

#pragma once

#include "armor.hpp"
#include "lightbar.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

// ==================== 可视化模块：把检测/配对结果画到图上 ====================
// 约定：所有函数只向传入的 img 上绘制，不修改原始图像数据（调用方自行保证传入副本）；
//       模块不依赖 iostream，不做任何控制台输出。

// 绘制单根灯条：蓝灯条画蓝框、红灯条画红框，中心画绿色实心点
void drawLightbars(cv::Mat &img, const std::vector<Lightbar> &lightbars);

// 绘制已配对的装甲板：两灯条四个端点连成绿色四边形，中心画点并标注 "Armor N"；
// 无配对时在左下角输出 "No armor detected" 提示
void drawArmors(cv::Mat &img, const std::vector<Armor> &armors);

#pragma once

#include "armor.hpp"
#include "pnp_solver.hpp"

#include <vector>

// 输出单张图片的灯条与装甲板检测详情
void printImageDetails(const std::vector<Lightbar> &lightbars,
                       const std::vector<Armor> &armors,
                       const ArmorMatcher &matcher);

// 输出视频处理过程中的单帧摘要
void printVideoSummary(int frame_index, int total_frames,
                       const std::vector<Lightbar> &lightbars,
                       const std::vector<Armor> &armors);

// 输出单张图片中每块装甲板的 PnP 位姿、相机坐标系下中心坐标 与 重投影误差
void printPnPResults(const std::vector<Armor> &armors,
                     const std::vector<PnPResult> &results);

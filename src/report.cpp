#include "report.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

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

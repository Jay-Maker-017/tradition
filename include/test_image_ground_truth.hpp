#pragma once

#include "pnp_solver.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct TestArmorGroundTruth
{
    std::string name;
    bool front_visible = false;
    cv::Vec3d tvec_m{0.0, 0.0, 0.0};
    cv::Vec3d rpy_deg{0.0, 0.0, 0.0};
    int width_axis = 0; // 0: local X, 1: local Y; height is local Z.
};

struct TestImageGroundTruth
{
    std::string image_filename;
    std::vector<TestArmorGroundTruth> armors;
};

struct ProjectedTruthArmor
{
    TestArmorGroundTruth truth;
    cv::Point2f center;
    std::vector<cv::Point2f> corners;
};

// The three records reproduce the camera-frame /tf values in test_imgs/图1-图3.md.
const TestImageGroundTruth *findTestImageGroundTruth(const std::string &image_path);

// Uses the small-armor dimensions (135 x 55 mm); the md files do not state plate size.
std::vector<ProjectedTruthArmor> projectTestImageGroundTruth(
    const TestImageGroundTruth &truth, const CameraParams &camera_params);

void drawTestImageGroundTruth(cv::Mat &image,
                              const std::vector<ProjectedTruthArmor> &projected);

void printTestImageCalibration(const TestImageGroundTruth &truth,
                               const std::vector<ProjectedTruthArmor> &projected,
                               const CameraParams &camera_params,
                               const cv::Size &image_size);

void printTestImagePoseErrors(const std::vector<ProjectedTruthArmor> &projected,
                              const std::vector<PnPResult> &estimates);

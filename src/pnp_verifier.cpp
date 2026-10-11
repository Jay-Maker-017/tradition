#include "pnp_verifier.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>

namespace
{
double wrappedAngleDifference(double estimate, double truth)
{
    return std::remainder(estimate - truth, 360.0);
}

double rotationDifferenceDegrees(const cv::Vec3d &estimate_rvec, const cv::Vec3d &truth_rvec)
{
    cv::Mat estimate_rotation;
    cv::Mat truth_rotation;
    cv::Rodrigues(estimate_rvec, estimate_rotation);
    cv::Rodrigues(truth_rvec, truth_rotation);
    const cv::Mat relative_rotation = estimate_rotation * truth_rotation.t();
    const double cosine = std::clamp((cv::trace(relative_rotation)[0] - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / CV_PI;
}
} // namespace

PoseError compareToGroundTruth(const PnPResult &estimate, const PoseGroundTruth &truth)
{
    PoseError error;
    if (!estimate.valid)
        return error;

    if (truth.has_distance)
    {
        error.has_distance_error = true;
        error.distance_error_m = estimate.distance_m - truth.distance_m;
    }
    if (truth.has_yaw)
    {
        error.has_yaw_error = true;
        error.yaw_error_deg = wrappedAngleDifference(estimate.yaw_deg, truth.yaw_deg);
    }
    if (truth.has_pitch)
    {
        error.has_pitch_error = true;
        error.pitch_error_deg = wrappedAngleDifference(estimate.pitch_deg, truth.pitch_deg);
    }
    if (truth.has_tvec)
    {
        error.has_tvec_error = true;
        error.tvec_error = estimate.tvec - truth.tvec;
    }
    if (truth.has_rvec)
    {
        error.has_rotation_error = true;
        error.rotation_error_deg = rotationDifferenceDegrees(estimate.rvec, truth.rvec);
    }
    return error;
}

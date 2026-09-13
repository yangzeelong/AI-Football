#pragma once

// --- Common headers ---
#include "common/Defer.hpp"
#include "common/MyMessage.hpp"
#include "common/PipelineCompletionSignal.hpp"

// --- Inference engine ---
#include "inference/IInferenceEngine.hpp"
#include "inference/TensorRTEngine.hpp"

// --- Detection ---
#include "detector/RFDetrDetectorInfer.hpp"
#include "detector/RFDetrDetector.hpp"

// --- Tracking ---
#include "tracker/ByteTracker.hpp"
#include "tracker/FootballTracker.hpp"

// --- Pose estimation ---
#include "pose/HRNetPoseEstimatorInfer.hpp"
#include "pose/HRNetPoseEstimator.hpp"

// --- Smoothing ---
#include "smoother/KeypointSmoother.hpp"

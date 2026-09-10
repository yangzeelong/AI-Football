#pragma once

// --- Common headers ---
#include "common/Defer.hpp"
#include "common/MyMessage.hpp"
#include "common/PipelineCompletionSignal.hpp"

// --- Inference engine ---
#include "inference/IInferenceEngine.hpp"
#include "inference/TensorRTEngine.hpp"

// --- Video input ---
#ifdef WITH_FFMPEG
#include "video/VideoReader.hpp"
#include "video/VideoDecoder.hpp"
#endif

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

// --- Output sinks ---
#include "output/ObservationWriter.hpp"
#include "output/AlarmPusher.hpp"

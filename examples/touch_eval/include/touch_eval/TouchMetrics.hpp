#pragma once

#include <touch_eval/TouchDetector.hpp>
#include <touch_eval/TouchGroundTruth.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace touch_eval {

/**
 * @file TouchMetrics.hpp
 * @brief Turns detected touches and the handwritten annotations into numbers.
 *
 * The annotations are coarse: a legacy .label line records "a touch happened in
 * this second", and it may count two touches in that second. Scoring that as
 * one point with one slot would punish the detector for the notation's
 * limitations, so the evaluator reports four layers:
 *
 *   1. count aware  - the primary metric. A point annotation is the window it
 *                     stands for (1 s for a .label second, 0.5 s for a 0.01 s
 *                     CSV instant) and it hands out `count` slots, so a line
 *                     reading "(2次)" can accept two detections instead of one.
 *   2. false positive breakdown - how much of the leftover is a repeated frame
 *                     of the same touch, an event next to an annotation, or a
 *                     genuinely unannotated detection.
 *   3. event level  - the previous one-to-one matching with a +- tolerance
 *                     sweep, kept so historical numbers stay comparable.
 *   4. attribution  - why each miss happened, plus the distributions that a
 *                     threshold has to separate.
 *
 * A fifth layer answers the other half of the question: the annotations flag the
 * touches whose frames are too degraded to judge (tiny players, a ball smeared
 * by motion blur). Those rows are scored twice, with and without them, because
 * "the detector missed something nobody can verify" and "the detector missed
 * something visible" are different claims.
 */

struct Percentiles {
    int samples = 0;
    double min = 0.0;
    double mean = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

/// Sorts the input in place and fills every percentile.
Percentiles ComputePercentiles(std::vector<double> values);

/// Metadata line of observations.jsonl.
struct ObservationMetadata {
    std::string videoPath;
    std::string cameraId;
    double fps = 0.0;
    int width = 0;
    int height = 0;
    int stride = 1;
};

struct ToleranceScore {
    double toleranceSec = 0.5;
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    /// Signed distance from the prediction to the annotated instant/window.
    Percentiles signedDtSec;
    Percentiles absDtSec;
};

/// Primary metric: what the annotation actually promises. A point annotation is
/// the window it stands for, and each annotation hands out `count` slots.
struct InstanceScore {
    /// Window a point annotation stands for (taken from the annotation source
    /// unless the CLI overrides it).
    double pointWindowSec = 1.0;
    /// Extra slack on both sides of every window.
    double windowPadSec = 0.0;
    /// Sum of the annotation counts: how many touches the file claims.
    int capacity = 0;
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    /// Signed distance from the event to the annotated instant (0 inside an
    /// interval annotation), for the timing report.
    Percentiles signedDtSec;
    Percentiles absDtSec;
    /// Annotations that received at least one / exactly `count` events.
    int windowsWithEvent = 0;
    int windowsExact = 0;
    /// Events the optional gates refused, so an over-eager gate is visible.
    int suppressedByRefractory = 0;
    int rejectedBySpeedGate = 0;
    int rejectedByBallGate = 0;
};

/// How one prediction relates to the annotations.
enum class EventVerdict {
    /// Inside an annotation window with a free slot: a true positive.
    Assigned = 0,
    /// Same person and foot as an assigned event, within `duplicateSec`: the
    /// same touch reported on another frame.
    Duplicate = 1,
    /// Inside a span the annotator confirmed has no touch.
    InsideNoTouch = 2,
    /// Inside an annotation window whose slots were already taken: the
    /// annotation says one touch happened there, the detector said two.
    InWindowNoSlot = 3,
    /// Outside every window but within `adjacencySec` of one.
    Adjacent = 4,
    /// Nothing annotated anywhere near: a genuine false detection.
    Unannotated = 5,
};

const char* EventVerdictName(EventVerdict verdict);

/// Where the false positives come from. Only `unannotated` is a real detector
/// error; the rest is the notation being coarser than the detector.
struct FalsePositiveBreakdown {
    /// Same person and foot as an event that was assigned, within
    /// `duplicateSec`: the same touch reported on another frame.
    int duplicates = 0;
    /// Inside a window the annotator confirmed has no touch.
    int insideNoTouch = 0;
    /// Inside an annotation window whose slots were already used up. Distinct
    /// from the neighbours below: the detector is not pointing at a different
    /// moment, it reported the same moment twice.
    int inWindowNoSlot = 0;
    /// Outside every window but within `adjacencySec` of one.
    int adjacent = 0;
    /// Nothing annotated anywhere near: a genuine false detection.
    int unannotated = 0;
    /// One verdict per predicted event, in the order of EvaluationResult::events.
    std::vector<EventVerdict> eventVerdict;
    std::vector<std::string> examples;
};

struct CountBucket {
    int gtIndex = 0;
    double startSec = 0.0;
    double endSec = 0.0;
    int gtCount = 0;
    int predictedCount = 0;
    bool exact = false;
};

struct CountScore {
    int records = 0;
    int exactRecords = 0;
    int overRecords = 0;
    int underRecords = 0;
    int gtTouches = 0;
    int predictedInWindows = 0;
    double meanAbsCountError = 0.0;
    std::vector<CountBucket> buckets;
};

struct FailureGroup {
    std::string reason;
    int count = 0;
    /// A few "gt#12 @ 1:46.00 (nearest 41.2 px)" style examples.
    std::vector<std::string> examples;
};

struct GtWindowDiagnostic {
    int records = 0;
    /// Windows that contained at least one ball detection (any state).
    int withDetectedBall = 0;
    /// Windows that contained at least one admissible (observed, confident) ball.
    int withUsableBall = 0;
    /// Windows that contained at least one usable foot keypoint.
    int withUsableFeet = 0;
    /// Distance from the ball to the closest foot, over the annotation windows.
    Percentiles nearestDistancePx;
    /// Same, restricted to frames with an admissible ball.
    Percentiles nearestUsableDistancePx;
    Percentiles personHeightPx;
    /// The same distance over every frame of the video, as a reference for how
    /// separable annotated touches are from the rest.
    Percentiles backgroundNearestDistancePx;
    Percentiles backgroundNearestUsableDistancePx;
};

struct ThresholdSweepPoint {
    double contactDistanceRatio = 0.0;
    /// Enter threshold that ratio produces for a 1280x720 frame and a 600 px
    /// tall player.
    double thresholdPxRef = 0.0;
    double toleranceSec = 0.0;
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    /// The same point under the count aware (primary) metric.
    double instancePrecision = 0.0;
    double instanceRecall = 0.0;
    double instanceF1 = 0.0;
};

struct EvaluationOptions {
    TouchDetectorParams params;
    /// Event matching tolerances, scanned in order. The annotations are
    /// second-resolution, so a fraction of a second is the tightest tolerance
    /// that can still be satisfied.
    std::vector<double> tolerances = {0.25, 0.5, 1.0, 2.0};
    /// Tolerance used for the auxiliary one-to-one numbers and count buckets.
    double primaryToleranceSec = 0.5;
    /// contactDistanceRatio values for the calibration curve.
    std::vector<double> sweepRatios = {0.005, 0.01, 0.015, 0.02, 0.03,
                                       0.045, 0.06, 0.08,  0.12, 0.2};
    /// Extra padding around an annotation window when looking at the frames.
    double windowSec = 0.5;
    /// Window a point annotation stands for. 0 = take it from the annotation
    /// source (.label 1.0 s, v2 CSV 0.5 s).
    double pointWindowSec = 0.0;
    /// Slack added on both sides of every window. Human timing error is real,
    /// but a large pad would hide missed touches, so the default is strict.
    double windowPadSec = 0.0;
    /// Distance below which an unassigned event is blamed on the notation
    /// ("the annotator wrote the neighbouring second") rather than on the
    /// detector.
    double adjacencySec = 1.5;
    /// Same person and foot within this many seconds = the same touch.
    double duplicateSec = 0.5;
};

//// The same predictions scored against the annotations with every row flagged
/// `low_quality_data` removed.
///
/// A flagged row is not evidence against the detector: nobody can say whether
/// the ball met the foot in those frames. Predictions that only a flagged
/// annotation could explain are dropped from both sides - neither a hit nor a
/// false positive - and `excludedEvents` reports how many that was, so the
/// second score cannot look better by quietly shrinking the test.
struct LowQualityScore {
    /// Rows cut by the flag: touches plus negative spans.
    int rows = 0;
    /// Touches those rows claimed.
    int touches = 0;
    /// Predictions dropped with them.
    int excludedEvents = 0;
    InstanceScore instance;
    std::vector<ToleranceScore> tolerances;
    ToleranceScore primary;
    CountScore counts;
    /// What the flagged windows themselves contained - the "why" behind the
    /// flag, and the check that the annotation is not flagging a scene that was
    /// perfectly readable.
    int windowsWithUsableBall = 0;
    int windowsWithUsableFeet = 0;
    Percentiles personHeightPx;
    Percentiles nearestUsableDistancePx;
};

// Everything the report needs, in one place.
struct EvaluationResult {
    std::size_t frames = 0;
    double durationSec = 0.0;
    int gtEvents = 0;
    int gtTouches = 0;
    int predictedEvents = 0;
    /// Time span the observations actually cover.
    double coveredFromSec = 0.0;
    double coveredToSec = 0.0;
    /// Annotations whose window overlaps the covered span. A short test run
    /// (--max_frames) cannot score the annotations beyond it, and recall would
    /// read as if they were all missed.
    int gtEventsCovered = 0;
    bool partialRun = false;
    /// Primary metric.
    InstanceScore instance;
    /// Second view of the very same run: the flagged rows are gone. `rows == 0`
    /// when the annotation flags nothing, and then the report says so instead of
    /// printing an empty table.
    LowQualityScore lowQuality;
    FalsePositiveBreakdown falsePositives;
    /// Auxiliary one-to-one numbers, kept comparable with earlier runs.
    std::vector<ToleranceScore> tolerances;
    ToleranceScore primary;
    CountScore counts;
    GtWindowDiagnostic diagnostics;
    std::vector<FailureGroup> failures;
    std::vector<ThresholdSweepPoint> sweep;
    int footHintRecords = 0;
    int footHintCorrect = 0;

    /// Predicted events in frame order.
    std::vector<TouchEvent> events;
    /// Primary assignment: annotation index, or -1 when the event is a false
    /// positive. Reported per event by touch_events.jsonl.
    std::vector<int> eventGtIndex;
    /// Per-frame contact signal of the primary run, for the CSV export.
    std::vector<FrameContact> contacts;
};

/// Replays the cached per-frame evidence and scores it against the annotation.
EvaluationResult EvaluateFrames(const std::vector<FrameEvidence>& frames,
                                const GroundTruth& groundTruth,
                                const EvaluationOptions& options);

/// Fills the default tolerances and sweep ratios.
EvaluationOptions DefaultEvaluationOptions();

struct ReportContext {
    ObservationMetadata metadata;
    std::string observationsPath;
    std::string groundTruthSource;
    EvaluationOptions options;
    GroundTruth groundTruth;
    /// True when the annotation was read from the dumped JSON.
    bool groundTruthFromJson = false;
};

std::string BuildReportText(const EvaluationResult& result,
                            const ReportContext& context);
bool WriteReportJson(const EvaluationResult& result,
                     const ReportContext& context, const std::string& path,
                     std::string* error);

} // namespace touch_eval

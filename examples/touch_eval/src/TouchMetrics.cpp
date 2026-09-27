#include <touch_eval/TouchMetrics.hpp>

#include <json/json.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace touch_eval {
namespace {

const double kInfinity = std::numeric_limits<double>::infinity();

double AbsValue(double value) { return value < 0.0 ? -value : value; }

/// Signed distance from a prediction to an annotation: negative when the
/// prediction came before the annotated instant/window, zero inside it.
double SignedDt(const GroundTruthEvent& groundTruth, double timeSec) {
    if (timeSec < groundTruth.startSec) return timeSec - groundTruth.startSec;
    if (timeSec > groundTruth.endSec) return timeSec - groundTruth.endSec;
    return 0.0;
}

struct MatchOutcome {
    int truePositives = 0;
    /// Per predicted event: matched annotation index, or -1 false positive.
    std::vector<int> eventGt;
    /// Per annotation: matched event index, or -1 false negative.
    std::vector<int> gtEvent;
    std::vector<double> signedDtSec;
};

MatchOutcome MatchEvents(const GroundTruth& groundTruth,
                         const std::vector<TouchEvent>& events,
                         double toleranceSec) {
    MatchOutcome outcome;
    outcome.eventGt.assign(events.size(), -1);
    outcome.gtEvent.assign(groundTruth.events.size(), -1);

    struct Pair {
        double absDtSec = 0.0;
        double signedDtSec = 0.0;
        int gtIndex = 0;
        int eventIndex = 0;
    };
    std::vector<Pair> pairs;
    for (std::size_t g = 0; g < groundTruth.events.size(); ++g) {
        for (std::size_t e = 0; e < events.size(); ++e) {
            const Pair candidate = [&]() {
                Pair value;
                value.signedDtSec =
                    SignedDt(groundTruth.events[g], events[e].timestampSec);
                value.absDtSec = AbsValue(value.signedDtSec);
                value.gtIndex = static_cast<int>(g);
                value.eventIndex = static_cast<int>(e);
                return value;
            }();
            if (candidate.absDtSec > toleranceSec) continue;
            pairs.push_back(candidate);
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& lhs, const Pair& rhs) {
        if (lhs.absDtSec != rhs.absDtSec) return lhs.absDtSec < rhs.absDtSec;
        if (lhs.gtIndex != rhs.gtIndex) return lhs.gtIndex < rhs.gtIndex;
        return lhs.eventIndex < rhs.eventIndex;
    });

    // Greedy one-to-one assignment, closest pairs first.
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const Pair& pair = pairs[i];
        if (outcome.eventGt[pair.eventIndex] != -1) continue;
        if (outcome.gtEvent[pair.gtIndex] != -1) continue;
        outcome.eventGt[pair.eventIndex] = pair.gtIndex;
        outcome.gtEvent[pair.gtIndex] = pair.eventIndex;
        outcome.signedDtSec.push_back(pair.signedDtSec);
    }
    outcome.truePositives = static_cast<int>(outcome.signedDtSec.size());
    return outcome;
}

/// The span of time a single annotation stands for.
struct GtWindow {
    double fromSec = 0.0;
    double toSec = 0.0;
};

GtWindow WindowOf(const GroundTruthEvent& groundTruth, double pointWindowSec,
                  double padSec) {
    GtWindow window;
    if (groundTruth.IsInterval()) {
        // A touch process the annotator watched from start to end.
        window.fromSec = groundTruth.startSec - padSec;
        window.toSec = groundTruth.endSec + padSec;
        return window;
    }
    // A point annotation is centred on the instant it names: the annotator's
    // own reaction lag is as likely to be early as late, and a centred window
    // makes a 1 s .label second behave exactly like the +-0.5 s tolerance of
    // the auxiliary metric.
    const double half = 0.5 * pointWindowSec;
    window.fromSec = groundTruth.startSec - half - padSec;
    window.toSec = groundTruth.startSec + half + padSec;
    return window;
}

bool WithinWindow(const GtWindow& window, double timeSec) {
    return timeSec >= window.fromSec && timeSec <= window.toSec;
}

/// Signed distance from the event to the annotated instant: negative before it.
/// For an interval annotation, zero when the event is anywhere inside.
double SignedOffset(const GroundTruthEvent& groundTruth, double timeSec) {
    if (groundTruth.IsInterval()) {
        if (timeSec < groundTruth.startSec) return timeSec - groundTruth.startSec;
        if (timeSec > groundTruth.endSec) return timeSec - groundTruth.endSec;
        return 0.0;
    }
    return timeSec - groundTruth.startSec;
}

struct InstanceMatch {
    int truePositives = 0;
    std::vector<int> eventGt;
    std::vector<int> gtAssigned;
    std::vector<double> signedDtSec;
};

/// Count aware assignment: an annotation may absorb up to `count` events that
/// fall inside the window it stands for. Events are taken closest-first, so the
/// slot of a crowded window goes to its most plausible detection.
InstanceMatch MatchInstances(const GroundTruth& groundTruth,
                             const std::vector<TouchEvent>& events,
                             double pointWindowSec, double padSec) {
    InstanceMatch match;
    match.eventGt.assign(events.size(), -1);
    match.gtAssigned.assign(groundTruth.events.size(), 0);

    struct Candidate {
        double absOffsetSec = 0.0;
        long long frameIndex = 0;
        int gtIndex = 0;
        int eventIndex = 0;
    };
    std::vector<Candidate> candidates;
    for (std::size_t g = 0; g < groundTruth.events.size(); ++g) {
        const GtWindow window =
            WindowOf(groundTruth.events[g], pointWindowSec, padSec);
        for (std::size_t e = 0; e < events.size(); ++e) {
            if (!WithinWindow(window, events[e].timestampSec)) continue;
            Candidate candidate;
            candidate.absOffsetSec = std::fabs(
                SignedOffset(groundTruth.events[g], events[e].timestampSec));
            candidate.frameIndex = static_cast<long long>(events[e].frameIndex);
            candidate.gtIndex = static_cast<int>(g);
            candidate.eventIndex = static_cast<int>(e);
            candidates.push_back(candidate);
        }
    }
    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& lhs, const Candidate& rhs) {
            if (lhs.absOffsetSec != rhs.absOffsetSec) {
                return lhs.absOffsetSec < rhs.absOffsetSec;
            }
            if (lhs.gtIndex != rhs.gtIndex) return lhs.gtIndex < rhs.gtIndex;
            return lhs.frameIndex < rhs.frameIndex;
        });

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Candidate& candidate = candidates[i];
        if (match.eventGt[candidate.eventIndex] != -1) continue;
        const int capacity =
            std::max(1, groundTruth.events[candidate.gtIndex].count);
        if (match.gtAssigned[candidate.gtIndex] >= capacity) continue;
        match.eventGt[candidate.eventIndex] = candidate.gtIndex;
        ++match.gtAssigned[candidate.gtIndex];
        ++match.truePositives;
        match.signedDtSec.push_back(SignedOffset(
            groundTruth.events[candidate.gtIndex],
            events[candidate.eventIndex].timestampSec));
    }
    return match;
}

/// Window a point annotation stands for: the CLI wins, the annotation source
/// decides otherwise. Every window built in this file goes through it, so the
/// matching and the false positive classification cannot disagree about what a
/// row means.
double EffectivePointWindow(const GroundTruth& groundTruth,
                            const EvaluationOptions& options) {
    return options.pointWindowSec > 0.0 ? options.pointWindowSec
                                        : groundTruth.defaultPointWindowSec;
}

/// The annotations without the rows the annotator flagged as too degraded to
/// judge.
GroundTruth WithoutLowQuality(const GroundTruth& groundTruth) {
    GroundTruth clean;
    clean.source = groundTruth.source;
    clean.sourceKind = groundTruth.sourceKind;
    clean.fps = groundTruth.fps;
    clean.defaultPointWindowSec = groundTruth.defaultPointWindowSec;
    clean.warnings = groundTruth.warnings;
    for (std::size_t i = 0; i < groundTruth.events.size(); ++i) {
        if (!groundTruth.events[i].lowQuality) {
            clean.events.push_back(groundTruth.events[i]);
        }
    }
    for (std::size_t i = 0; i < groundTruth.noTouch.size(); ++i) {
        if (!groundTruth.noTouch[i].lowQuality) {
            clean.noTouch.push_back(groundTruth.noTouch[i]);
        }
    }
    return clean;
}

/// Predictions whose only possible explanation is a flagged annotation: they
/// sit in the window that row stands for, so scoring them as a false positive
/// would blame the detector for frames nobody can read.
std::vector<TouchEvent> DropLowQualityEvents(
    const GroundTruth& groundTruth, const std::vector<TouchEvent>& events,
    const EvaluationOptions& options, int* excluded) {
    const double pointWindowSec = EffectivePointWindow(groundTruth, options);
    std::vector<GtWindow> windows;
    for (std::size_t i = 0; i < groundTruth.events.size(); ++i) {
        if (!groundTruth.events[i].lowQuality) continue;
        windows.push_back(WindowOf(groundTruth.events[i], pointWindowSec,
                                   options.windowPadSec));
    }
    for (std::size_t i = 0; i < groundTruth.noTouch.size(); ++i) {
        if (!groundTruth.noTouch[i].lowQuality) continue;
        windows.push_back(WindowOf(groundTruth.noTouch[i], pointWindowSec,
                                   options.windowPadSec));
    }

    std::vector<TouchEvent> kept;
    kept.reserve(events.size());
    for (std::size_t e = 0; e < events.size(); ++e) {
        bool dropped = false;
        for (std::size_t w = 0; w < windows.size() && !dropped; ++w) {
            dropped = WithinWindow(windows[w], events[e].timestampSec);
        }
        if (dropped) {
            if (excluded) ++(*excluded);
        } else {
            kept.push_back(events[e]);
        }
    }
    return kept;
}

InstanceScore ScoreInstances(const GroundTruth& groundTruth,
                             const std::vector<TouchEvent>& events,
                             const EvaluationOptions& options) {
    InstanceScore score;
    score.pointWindowSec = EffectivePointWindow(groundTruth, options);
    score.windowPadSec = options.windowPadSec;
    score.capacity = groundTruth.TouchCount();

    const InstanceMatch match =
        MatchInstances(groundTruth, events, score.pointWindowSec,
                       score.windowPadSec);
    score.truePositives = match.truePositives;
    score.falsePositives = static_cast<int>(events.size()) - match.truePositives;
    score.falseNegatives = score.capacity - match.truePositives;
    if (score.truePositives + score.falsePositives > 0) {
        score.precision = static_cast<double>(score.truePositives) /
                          (score.truePositives + score.falsePositives);
    }
    if (score.truePositives + score.falseNegatives > 0) {
        score.recall = static_cast<double>(score.truePositives) /
                       (score.truePositives + score.falseNegatives);
    }
    if (score.precision + score.recall > 0.0) {
        score.f1 = 2.0 * score.precision * score.recall /
                   (score.precision + score.recall);
    }
    for (std::size_t g = 0; g < match.gtAssigned.size(); ++g) {
        if (match.gtAssigned[g] > 0) ++score.windowsWithEvent;
        if (match.gtAssigned[g] ==
            std::max(1, groundTruth.events[g].count)) {
            ++score.windowsExact;
        }
    }
    std::vector<double> signedValues;
    std::vector<double> absValues;
    for (std::size_t i = 0; i < match.signedDtSec.size(); ++i) {
        signedValues.push_back(match.signedDtSec[i]);
        absValues.push_back(std::fabs(match.signedDtSec[i]));
    }
    score.signedDtSec = ComputePercentiles(signedValues);
    score.absDtSec = ComputePercentiles(absValues);
    return score;
}

std::string DescribeEvent(const TouchEvent& event) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(2);
    out << "frame " << event.frameIndex << " @ " << FormatSeconds(event.timestampSec)
        << " p" << event.personTrackId << (event.leftFoot ? " left" : " right")
        << " d=" << event.distancePx << "px";
    return out.str();
}

/// Splits the leftover detections by where they sit relative to the
/// annotations, so "the notation is coarse" and "the detector made it up" are
/// not added together.
FalsePositiveBreakdown ClassifyFalsePositives(
    const GroundTruth& groundTruth, const std::vector<TouchEvent>& events,
    const std::vector<int>& eventGt, const EvaluationOptions& options) {
    FalsePositiveBreakdown breakdown;
    breakdown.eventVerdict.assign(events.size(), EventVerdict::Assigned);
    // The same window the assignment used: reading `options.pointWindowSec`
    // directly would collapse a point annotation to zero width whenever the CLI
    // leaves the window at "take it from the source", and then "next to an
    // annotation" would be measured from the instant instead of from the window
    // edge.
    const double pointWindowSec = EffectivePointWindow(groundTruth, options);
    std::vector<GtWindow> windows;
    windows.reserve(groundTruth.events.size() + groundTruth.noTouch.size());
    for (std::size_t g = 0; g < groundTruth.events.size(); ++g) {
        windows.push_back(WindowOf(groundTruth.events[g], pointWindowSec,
                                   options.windowPadSec));
    }

    for (std::size_t e = 0; e < events.size(); ++e) {
        if (eventGt[e] >= 0) continue;
        const TouchEvent& event = events[e];
        std::string reason;

        bool insideNoTouch = false;
        for (std::size_t n = 0; n < groundTruth.noTouch.size(); ++n) {
            const GtWindow window = WindowOf(groundTruth.noTouch[n],
                                             pointWindowSec,
                                             options.windowPadSec);
            if (WithinWindow(window, event.timestampSec)) {
                insideNoTouch = true;
                break;
            }
        }
        bool duplicate = false;
        double nearestAssignedSec = 0.0;
        for (std::size_t other = 0; other < events.size() && !duplicate; ++other) {
            if (eventGt[other] < 0) continue;
            if (events[other].personTrackId != event.personTrackId) continue;
            if (events[other].leftFoot != event.leftFoot) continue;
            const double distance = std::fabs(events[other].timestampSec -
                                              event.timestampSec);
            if (distance <= options.duplicateSec) {
                duplicate = true;
                nearestAssignedSec = distance;
            }
        }
        double nearestWindowSec = kInfinity;
        for (std::size_t g = 0; g < windows.size(); ++g) {
            const double distance = [&windows, g, &event]() {
                if (WithinWindow(windows[g], event.timestampSec)) return 0.0;
                const double before = windows[g].fromSec - event.timestampSec;
                const double after = event.timestampSec - windows[g].toSec;
                return before > 0.0 ? before : after;
            }();
            nearestWindowSec = std::min(nearestWindowSec, distance);
        }

        std::ostringstream example;
        example << DescribeEvent(event);
        if (insideNoTouch) {
            ++breakdown.insideNoTouch;
            breakdown.eventVerdict[e] = EventVerdict::InsideNoTouch;
            reason = "inside a no_touch span";
            example << " - " << reason;
        } else if (duplicate) {
            ++breakdown.duplicates;
            breakdown.eventVerdict[e] = EventVerdict::Duplicate;
            reason = "duplicate";
            example.setf(std::ios::fixed);
            example.precision(2);
            example << " - same person and foot as an assigned event "
                    << nearestAssignedSec << " s away";
        } else if (nearestWindowSec == 0.0) {
            // Inside a window whose slots were already handed out. Reported
            // separately from the neighbours below: the detector is not looking
            // at a different moment, it took one touch for two.
            ++breakdown.inWindowNoSlot;
            breakdown.eventVerdict[e] = EventVerdict::InWindowNoSlot;
            reason = "inside an annotation window, but every slot was taken";
            example << " - " << reason;
        } else if (nearestWindowSec <= options.adjacencySec) {
            ++breakdown.adjacent;
            breakdown.eventVerdict[e] = EventVerdict::Adjacent;
            example.setf(std::ios::fixed);
            example.precision(2);
            example << " - " << nearestWindowSec << " s outside the nearest window";
        } else {
            ++breakdown.unannotated;
            breakdown.eventVerdict[e] = EventVerdict::Unannotated;
            example << " - no annotation within "
                    << options.adjacencySec << " s";
        }
        if (breakdown.examples.size() < 24) breakdown.examples.push_back(example.str());
    }
    return breakdown;
}

struct ReplayResult {
    std::vector<TouchEvent> events;
    std::vector<FrameContact> contacts;
    bool hasContacts = false;
    int suppressedByRefractory = 0;
    int rejectedBySpeedGate = 0;
    int rejectedByBallGate = 0;
};

/// Replays the cached evidence through the very same detector the online app
/// uses, so a gate cannot behave differently in the report and on screen.
ReplayResult Replay(const std::vector<FrameEvidence>& frames,
                    const TouchDetectorParams& params, bool keepContacts) {
    ReplayResult out;
    out.hasContacts = keepContacts;
    if (keepContacts) out.contacts.reserve(frames.size());

    TouchDetector detector(params);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const std::vector<TouchEvent> emitted = detector.PushEvidence(frames[i]);
        out.events.insert(out.events.end(), emitted.begin(), emitted.end());
        if (keepContacts) out.contacts.push_back(detector.LastContact());
    }
    const std::vector<TouchEvent> flushed = detector.Flush();
    out.events.insert(out.events.end(), flushed.begin(), flushed.end());
    out.suppressedByRefractory = detector.SuppressedByRefractory();
    out.rejectedBySpeedGate = detector.Aggregator().RejectedBySpeedGate();
    out.rejectedByBallGate = detector.Aggregator().RejectedByBallGate();
    return out;
}

ToleranceScore ScoreTolerance(const GroundTruth& groundTruth,
                              const std::vector<TouchEvent>& events,
                              double toleranceSec) {
    const MatchOutcome match = MatchEvents(groundTruth, events, toleranceSec);
    ToleranceScore score;
    score.toleranceSec = toleranceSec;
    score.truePositives = match.truePositives;
    score.falsePositives =
        static_cast<int>(events.size()) - match.truePositives;
    score.falseNegatives =
        static_cast<int>(groundTruth.events.size()) - match.truePositives;
    if (score.truePositives + score.falsePositives > 0) {
        score.precision = static_cast<double>(score.truePositives) /
                          (score.truePositives + score.falsePositives);
    }
    if (score.truePositives + score.falseNegatives > 0) {
        score.recall = static_cast<double>(score.truePositives) /
                       (score.truePositives + score.falseNegatives);
    }
    if (score.precision + score.recall > 0.0) {
        score.f1 = 2.0 * score.precision * score.recall /
                   (score.precision + score.recall);
    }
    std::vector<double> signedValues;
    std::vector<double> absValues;
    signedValues.reserve(match.signedDtSec.size());
    absValues.reserve(match.signedDtSec.size());
    for (std::size_t i = 0; i < match.signedDtSec.size(); ++i) {
        signedValues.push_back(match.signedDtSec[i]);
        absValues.push_back(AbsValue(match.signedDtSec[i]));
    }
    score.signedDtSec = ComputePercentiles(signedValues);
    score.absDtSec = ComputePercentiles(absValues);
    return score;
}

CountScore ScoreCounts(const GroundTruth& groundTruth,
                       const std::vector<TouchEvent>& events,
                       double toleranceSec) {
    CountScore score;
    score.records = static_cast<int>(groundTruth.events.size());
    score.gtTouches = groundTruth.TouchCount();

    for (std::size_t g = 0; g < groundTruth.events.size(); ++g) {
        const GroundTruthEvent& gt = groundTruth.events[g];
        CountBucket bucket;
        bucket.gtIndex = gt.index;
        bucket.startSec = gt.startSec;
        bucket.endSec = gt.endSec;
        bucket.gtCount = gt.count;

        const double from = gt.startSec - toleranceSec;
        const double to = gt.endSec + toleranceSec;
        for (std::size_t e = 0; e < events.size(); ++e) {
            if (events[e].timestampSec >= from && events[e].timestampSec <= to) {
                ++bucket.predictedCount;
            }
        }
        bucket.exact = (bucket.predictedCount == bucket.gtCount);
        if (bucket.exact) {
            ++score.exactRecords;
        } else if (bucket.predictedCount > bucket.gtCount) {
            ++score.overRecords;
        } else {
            ++score.underRecords;
        }
        score.predictedInWindows += bucket.predictedCount;
        score.meanAbsCountError +=
            AbsValue(static_cast<double>(bucket.predictedCount - bucket.gtCount));
        score.buckets.push_back(bucket);
    }
    if (!score.buckets.empty()) {
        score.meanAbsCountError /= static_cast<double>(score.buckets.size());
    }
    return score;
}

/// What the frames of one annotation window contained.
struct WindowStats {
    bool hasDetectedBall = false;
    bool hasUsableBall = false;
    bool hasUsableFeet = false;
    double nearestDistancePx = kInfinity;
    double nearestUsableDistancePx = kInfinity;
    double nearestPersonHeightPx = 0.0;
    int nearestFrameWidth = 0;
    int nearestFrameHeight = 0;
    double nearestTimestampSec = 0.0;
    double maxBallConfidence = 0.0;
    int frames = 0;
};

WindowStats ScanWindow(const std::vector<FrameEvidence>& frames,
                       const std::vector<double>& timestamps, double fromSec,
                       double toSec) {
    WindowStats stats;
    std::size_t begin = static_cast<std::size_t>(std::lower_bound(
        timestamps.begin(), timestamps.end(), fromSec) - timestamps.begin());
    for (std::size_t i = begin; i < frames.size(); ++i) {
        if (frames[i].timestampSec > toSec) break;
        const FrameEvidence& frame = frames[i];
        ++stats.frames;
        if (frame.ballCount > 0) stats.hasDetectedBall = true;
        if (frame.observedBallCount > 0) stats.hasUsableBall = true;
        if (frame.validFootPointCount > 0) stats.hasUsableFeet = true;
        if (frame.candidates.empty()) continue;

        const ContactCandidate& nearest = frame.candidates[0];
        if (nearest.distancePx < stats.nearestDistancePx) {
            stats.nearestDistancePx = nearest.distancePx;
            stats.nearestPersonHeightPx = nearest.personHeightPx;
            stats.nearestFrameWidth = frame.frameWidth;
            stats.nearestFrameHeight = frame.frameHeight;
            stats.nearestTimestampSec = frame.timestampSec;
        }
        if (frame.observedBallCount > 0 &&
            nearest.distancePx < stats.nearestUsableDistancePx) {
            stats.nearestUsableDistancePx = nearest.distancePx;
        }
        stats.maxBallConfidence =
            std::max(stats.maxBallConfidence, static_cast<double>(nearest.ballConfidence));
    }
    return stats;
}

void AddFailure(std::vector<FailureGroup>* groups, const std::string& reason,
                const std::string& example) {
    for (std::size_t i = 0; i < groups->size(); ++i) {
        if ((*groups)[i].reason != reason) continue;
        ++(*groups)[i].count;
        if ((*groups)[i].examples.size() < 8) {
            (*groups)[i].examples.push_back(example);
        }
        return;
    }
    FailureGroup group;
    group.reason = reason;
    group.count = 1;
    group.examples.push_back(example);
    groups->push_back(group);
}

std::string FormatValue(double value, int precision = 2) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(precision);
    out << value;
    return out.str();
}

std::string DescribePercentiles(const Percentiles& value, int precision = 2) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(precision);
    out << "n=" << value.samples << " mean=" << value.mean << " p50=" << value.p50
        << " p90=" << value.p90 << " p95=" << value.p95 << " max=" << value.max;
    return out.str();
}

std::string Percent(double value) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << value;
    return out.str();
}

} // namespace

Percentiles ComputePercentiles(std::vector<double> values) {
    Percentiles result;
    if (values.empty()) return result;
    std::sort(values.begin(), values.end());
    result.samples = static_cast<int>(values.size());
    result.min = values.front();
    result.max = values.back();
    double total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) total += values[i];
    result.mean = total / static_cast<double>(values.size());

    // Nearest-rank percentile, the same convention the pipeline timers use.
    const auto at = [&values](double fraction) {
        const double rank = fraction * static_cast<double>(values.size() - 1);
        std::size_t index = static_cast<std::size_t>(rank + 0.5);
        if (index >= values.size()) index = values.size() - 1;
        return values[index];
    };
    result.p50 = at(0.50);
    result.p90 = at(0.90);
    result.p95 = at(0.95);
    result.p99 = at(0.99);
    return result;
}

EvaluationOptions DefaultEvaluationOptions() { return EvaluationOptions(); }

const char* EventVerdictName(EventVerdict verdict) {
    switch (verdict) {
        case EventVerdict::Assigned: return "assigned";
        case EventVerdict::Duplicate: return "duplicate";
        case EventVerdict::InsideNoTouch: return "inside_no_touch";
        case EventVerdict::InWindowNoSlot: return "in_window_no_slot";
        case EventVerdict::Adjacent: return "adjacent";
        case EventVerdict::Unannotated: return "unannotated";
    }
    return "unannotated";
}

EvaluationResult EvaluateFrames(const std::vector<FrameEvidence>& frames,
                                const GroundTruth& groundTruth,
                                const EvaluationOptions& options) {
    EvaluationResult result;
    result.frames = frames.size();
    result.gtEvents = static_cast<int>(groundTruth.events.size());
    result.gtTouches = groundTruth.TouchCount();
    if (!frames.empty()) {
        result.durationSec = frames.back().timestampSec - frames.front().timestampSec;
        result.coveredFromSec = frames.front().timestampSec;
        result.coveredToSec = frames.back().timestampSec;
    }
    for (std::size_t g = 0; g < groundTruth.events.size(); ++g) {
        const GroundTruthEvent& gt = groundTruth.events[g];
        if (gt.endSec + options.primaryToleranceSec < result.coveredFromSec) continue;
        if (gt.startSec - options.primaryToleranceSec > result.coveredToSec) continue;
        ++result.gtEventsCovered;
    }
    result.partialRun = result.gtEventsCovered < result.gtEvents;

    // --- replay and the count aware primary metric -------------------------
    const ReplayResult replay = Replay(frames, options.params, true);
    result.events = replay.events;
    result.contacts = replay.contacts;
    result.predictedEvents = static_cast<int>(replay.events.size());

    result.instance = ScoreInstances(groundTruth, replay.events, options);
    result.instance.suppressedByRefractory = replay.suppressedByRefractory;
    result.instance.rejectedBySpeedGate = replay.rejectedBySpeedGate;
    result.instance.rejectedByBallGate = replay.rejectedByBallGate;
    const InstanceMatch instanceMatch = MatchInstances(
        groundTruth, replay.events, result.instance.pointWindowSec,
        result.instance.windowPadSec);
    result.eventGtIndex = instanceMatch.eventGt;
    result.falsePositives = ClassifyFalsePositives(
        groundTruth, replay.events, result.eventGtIndex, options);

    // --- auxiliary one-to-one numbers, comparable with earlier runs --------
    for (std::size_t i = 0; i < options.tolerances.size(); ++i) {
        result.tolerances.push_back(
            ScoreTolerance(groundTruth, replay.events, options.tolerances[i]));
    }
    result.primary = ScoreTolerance(groundTruth, replay.events,
                                    options.primaryToleranceSec);
    result.counts = ScoreCounts(groundTruth, replay.events,
                               options.primaryToleranceSec);

    // --- frame distributions and window diagnostics ------------------------
    std::vector<double> timestamps;
    timestamps.reserve(frames.size());
    std::vector<double> allNearest;
    std::vector<double> allNearestUsable;
    allNearest.reserve(frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) {
        timestamps.push_back(frames[i].timestampSec);
        if (frames[i].candidates.empty()) continue;
        allNearest.push_back(frames[i].candidates[0].distancePx);
        if (frames[i].observedBallCount > 0) {
            allNearestUsable.push_back(frames[i].candidates[0].distancePx);
        }
    }
    result.diagnostics.backgroundNearestDistancePx = ComputePercentiles(allNearest);
    result.diagnostics.backgroundNearestUsableDistancePx =
        ComputePercentiles(allNearestUsable);

    std::vector<double> windowNearest;
    std::vector<double> windowNearestUsable;
    std::vector<double> windowHeights;
    for (std::size_t g = 0; g < groundTruth.events.size(); ++g) {
        const GroundTruthEvent& gt = groundTruth.events[g];
        const WindowStats stats = ScanWindow(
            frames, timestamps, gt.startSec - options.windowSec,
            gt.endSec + options.windowSec);
        ++result.diagnostics.records;
        if (stats.hasDetectedBall) ++result.diagnostics.withDetectedBall;
        if (stats.hasUsableBall) ++result.diagnostics.withUsableBall;
        if (stats.hasUsableFeet) ++result.diagnostics.withUsableFeet;
        if (stats.nearestDistancePx < kInfinity) {
            windowNearest.push_back(stats.nearestDistancePx);
            windowHeights.push_back(stats.nearestPersonHeightPx);
        }
        if (stats.nearestUsableDistancePx < kInfinity) {
            windowNearestUsable.push_back(stats.nearestUsableDistancePx);
        }

        // Missed under the primary metric: no event landed in this window.
        if (instanceMatch.gtAssigned[g] > 0) continue;

        // --- why was this touch missed? ------------------------------------
        std::ostringstream example;
        example.setf(std::ios::fixed);
        example.precision(2);
        example << "gt#" << gt.index << " @ " << FormatSeconds(gt.startSec);
        if (gt.IsInterval()) {
            example << "-" << FormatSeconds(gt.endSec);
        }
        example << " (window " << FormatSeconds(gt.startSec - options.windowSec)
                << "-" << FormatSeconds(gt.endSec + options.windowSec) << ")";

        std::string reason;
        if (!stats.hasDetectedBall) {
            reason = "no ball detected in the window";
        } else if (!stats.hasUsableBall) {
            reason = "ball detected but never admissible (extrapolated or low confidence)";
        } else if (!stats.hasUsableFeet) {
            reason = "no usable foot keypoints (missing or interpolated)";
            example << ", nearest " << FormatValue(stats.nearestDistancePx)
                    << " px";
        } else {
            const float threshold = ContactThresholdPx(
                options.params, stats.nearestFrameWidth, stats.nearestFrameHeight,
                static_cast<float>(stats.nearestPersonHeightPx));
            if (stats.nearestUsableDistancePx <= threshold) {
                reason = "contact frames present but no accepted event (too short or merged)";
                example << ", nearest " << FormatValue(stats.nearestUsableDistancePx)
                        << " px <= tau " << FormatValue(threshold) << " px";
            } else {
                reason = "ball and feet present but never closer than the threshold";
                example << ", nearest " << FormatValue(stats.nearestUsableDistancePx)
                        << " px > tau " << FormatValue(threshold) << " px";
            }
        }
        AddFailure(&result.failures, reason, example.str());
    }
    result.diagnostics.nearestDistancePx = ComputePercentiles(windowNearest);
    result.diagnostics.nearestUsableDistancePx =
        ComputePercentiles(windowNearestUsable);
    result.diagnostics.personHeightPx = ComputePercentiles(windowHeights);

    // False positives, grouped so they can be reviewed by hand.
    for (std::size_t e = 0; e < replay.events.size(); ++e) {
        if (result.eventGtIndex[e] != -1) continue;
        const TouchEvent& event = replay.events[e];
        std::ostringstream example;
        example.setf(std::ios::fixed);
        example.precision(2);
        example << "event frame " << event.frameIndex << " @ "
                << FormatSeconds(event.timestampSec) << " person "
                << event.personTrackId << (event.leftFoot ? " left" : " right")
                << " d=" << event.distancePx << " px";
        AddFailure(&result.failures, "false positive (outside every annotation window)",
                   example.str());
    }

    // --- foot hint check (only a few annotations carry one) ----------------
    for (std::size_t g = 0; g < groundTruth.events.size(); ++g) {
        const GroundTruthEvent& gt = groundTruth.events[g];
        if (gt.footHint.empty()) continue;
        ++result.footHintRecords;
        const double from = gt.startSec - options.primaryToleranceSec;
        const double to = gt.endSec + options.primaryToleranceSec;
        bool sawLeft = false;
        bool sawRight = false;
        for (std::size_t e = 0; e < replay.events.size(); ++e) {
            const double time = replay.events[e].timestampSec;
            if (time < from || time > to) continue;
            if (replay.events[e].leftFoot) {
                sawLeft = true;
            } else {
                sawRight = true;
            }
        }
        const bool correct = (gt.footHint == "left") ? sawLeft
                             : (gt.footHint == "right") ? sawRight
                                                        : (sawLeft && sawRight);
        if (correct) ++result.footHintCorrect;
    }

    // --- threshold calibration sweep ---------------------------------------
    for (std::size_t i = 0; i < options.sweepRatios.size(); ++i) {
        TouchDetectorParams sweepParams = options.params;
        sweepParams.contactDistanceRatio =
            static_cast<float>(options.sweepRatios[i]);
        const ReplayResult sweepReplay = Replay(frames, sweepParams, false);

        const ToleranceScore score = ScoreTolerance(
            groundTruth, sweepReplay.events, options.primaryToleranceSec);
        const InstanceScore instanceScore =
            ScoreInstances(groundTruth, sweepReplay.events, options);
        ThresholdSweepPoint point;
        point.contactDistanceRatio = options.sweepRatios[i];
        // Frame level enter threshold for a 1280x720 source: the sweep only
        // moves the ratio, so this is what the ratio actually changes. A
        // distant player's own height cap can still lower it per person.
        point.thresholdPxRef =
            ContactThresholdPx(sweepParams, 1280, 720, 1.0e6f);
        point.toleranceSec = options.primaryToleranceSec;
        point.truePositives = score.truePositives;
        point.falsePositives = score.falsePositives;
        point.falseNegatives = score.falseNegatives;
        point.precision = score.precision;
        point.recall = score.recall;
        point.f1 = score.f1;
        point.instancePrecision = instanceScore.precision;
        point.instanceRecall = instanceScore.recall;
        point.instanceF1 = instanceScore.f1;
        result.sweep.push_back(point);
    }

    // --- the same run with the unreadable rows taken out --------------------
    // Nothing is replayed here: the detections are the same, only the reference
    // they are scored against changes, which is exactly what makes the two
    // numbers comparable.
    if (groundTruth.LowQualityRows() > 0) {
        result.lowQuality.rows = groundTruth.LowQualityRows();
        result.lowQuality.touches = groundTruth.LowQualityTouchCount();
        const GroundTruth cleanGroundTruth = WithoutLowQuality(groundTruth);
        const std::vector<TouchEvent> cleanEvents = DropLowQualityEvents(
            groundTruth, replay.events, options,
            &result.lowQuality.excludedEvents);

        result.lowQuality.instance =
            ScoreInstances(cleanGroundTruth, cleanEvents, options);
        result.lowQuality.instance.suppressedByRefractory =
            replay.suppressedByRefractory;
        result.lowQuality.instance.rejectedBySpeedGate =
            replay.rejectedBySpeedGate;
        result.lowQuality.instance.rejectedByBallGate = replay.rejectedByBallGate;
        for (std::size_t i = 0; i < options.tolerances.size(); ++i) {
            result.lowQuality.tolerances.push_back(ScoreTolerance(
                cleanGroundTruth, cleanEvents, options.tolerances[i]));
        }
        result.lowQuality.primary = ScoreTolerance(
            cleanGroundTruth, cleanEvents, options.primaryToleranceSec);
        result.lowQuality.counts = ScoreCounts(
            cleanGroundTruth, cleanEvents, options.primaryToleranceSec);

        // What the flagged windows looked like. A flag that turns out to sit on
        // a large player with a confident ball is a flag worth re-checking.
        std::vector<double> flaggedHeights;
        std::vector<double> flaggedNearest;
        const auto scanFlagged = [&](const std::vector<GroundTruthEvent>& rows) {
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const GroundTruthEvent& row = rows[i];
                if (!row.lowQuality) continue;
                const WindowStats stats =
                    ScanWindow(frames, timestamps, row.startSec - options.windowSec,
                               row.endSec + options.windowSec);
                if (stats.hasUsableBall) ++result.lowQuality.windowsWithUsableBall;
                if (stats.hasUsableFeet) ++result.lowQuality.windowsWithUsableFeet;
                if (stats.nearestDistancePx < kInfinity) {
                    flaggedHeights.push_back(stats.nearestPersonHeightPx);
                }
                if (stats.nearestUsableDistancePx < kInfinity) {
                    flaggedNearest.push_back(stats.nearestUsableDistancePx);
                }
            }
        };
        scanFlagged(groundTruth.events);
        scanFlagged(groundTruth.noTouch);
        result.lowQuality.personHeightPx = ComputePercentiles(flaggedHeights);
        result.lowQuality.nearestUsableDistancePx =
            ComputePercentiles(flaggedNearest);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

namespace {

void AppendFailureGroup(std::ostringstream& out, const FailureGroup& group) {
    out << "  [" << group.count << "] " << group.reason << "\n";
    for (std::size_t i = 0; i < group.examples.size(); ++i) {
        out << "        " << group.examples[i] << "\n";
    }
}

std::string FooterNote() {
    return std::string(
        "Notes:\n"
        "  - primary: a point annotation is the window it stands for (a .label\n"
        "    second, or a 0.01 s CSV instant) and it hands out `count` slots, so a\n"
        "    row reading (2次) can accept two detections. Anything else is a false\n"
        "    positive, classified further down by where it sits.\n"
        "  - auxiliary: one-to-one matching at +-tolerance, i.e. the metric used\n"
        "    before the count aware one existed; kept for comparability only.\n"
        "  - low_quality_data: a row whose frames are too degraded for the\n"
        "    annotator to judge. Those rows, and the detections only they could\n"
        "    explain, are removed for a second view; how far the two views differ\n"
        "    is how much of the score rests on unreadable frames.\n"
        "  - 'counts' compares, per annotation, how many touches were predicted in\n"
        "    its window under the auxiliary window definition.\n");
}

} // namespace

std::string BuildReportText(const EvaluationResult& result,
                            const ReportContext& context) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out << "===== touch evaluation =====\n";
    out << "observations   : " << context.observationsPath << "\n";
    out << "video          : " << context.metadata.videoPath << "\n";
    out << "video format   : " << context.metadata.width << "x"
        << context.metadata.height << " @ " << FormatValue(context.metadata.fps, 3)
        << " fps, stride " << context.metadata.stride << "\n";
    out << "frames         : " << result.frames << " (" << FormatValue(result.durationSec, 2)
        << " s)" << "\n";
    out << "ground truth   : " << context.groundTruthSource << " ("
        << context.groundTruth.sourceKind << ")\n";
    out << "gt events      : " << result.gtEvents << " annotations, "
        << result.gtTouches << " touches, "
        << context.groundTruth.noTouch.size() << " no_touch span(s)";
    if (result.lowQuality.rows > 0) {
        out << ", " << result.lowQuality.rows << " low_quality_data row(s)";
    }
    out << "\n";
    out << "predicted      : " << result.predictedEvents << " events\n";
    out << "params         : " << DescribeTouchDetectorParams(context.options.params)
        << "\n";
    if (result.partialRun) {
        out << "\n  !! PARTIAL RUN: the observations cover "
            << FormatSeconds(result.coveredFromSec) << "-"
            << FormatSeconds(result.coveredToSec) << " but only "
            << result.gtEventsCovered << " of " << result.gtEvents
            << " annotations fall inside that span. Recall and precision below\n"
               "     score the whole label, so they are not comparable with a "
               "full run.\n";
    }
    out << "\n";

    // --- primary metric ----------------------------------------------------
    out << "-- PRIMARY: count aware (each annotation = a window with `count` slots) --\n";
    out << "  point window   : " << FormatValue(result.instance.pointWindowSec, 2)
        << " s (source default "
        << FormatValue(context.groundTruth.defaultPointWindowSec, 2)
        << " s), pad " << FormatValue(result.instance.windowPadSec, 2) << " s; "
        << "interval annotations keep their own span\n";
    out << "  capacity       : " << result.instance.capacity
        << " touches claimed by the annotations\n";
    out << "  TP=" << result.instance.truePositives
        << " FP=" << result.instance.falsePositives
        << " FN=" << result.instance.falseNegatives
        << "  precision=" << Percent(result.instance.precision)
        << " recall=" << Percent(result.instance.recall)
        << " f1=" << Percent(result.instance.f1) << "\n";
    out << "  timing error vs the annotated instant (s): "
        << DescribePercentiles(result.instance.signedDtSec, 3) << "\n";
    out << "  |timing error| (s)                       : "
        << DescribePercentiles(result.instance.absDtSec, 3) << "\n";
    out << "  annotations hit / met exactly            : "
        << result.instance.windowsWithEvent << " / "
        << result.instance.windowsExact << " of " << result.gtEvents << "\n";
    if (result.instance.suppressedByRefractory > 0 ||
        result.instance.rejectedBySpeedGate > 0 ||
        result.instance.rejectedByBallGate > 0) {
        out << "  events refused by the gates              : "
            << result.instance.suppressedByRefractory << " refractory, "
            << result.instance.rejectedBySpeedGate << " speed, "
            << result.instance.rejectedByBallGate << " tracked-ball-only\n";
    }

    if (result.lowQuality.rows == 0) {
        out << "  low quality    : no row flagged, so the numbers above are the "
               "only view. Mark a \n"
               "                   row `yes` in low_quality_data to score a "
               "second view without it.\n";
    } else {
        const LowQualityScore& clean = result.lowQuality;
        out << "\n-- PRIMARY with the unreadable rows taken out --\n";
        out << "  flagged rows   : " << clean.rows << " row(s) / "
            << clean.touches << " touch(es) marked low_quality_data\n";
        out << "  dropped with them: " << clean.excludedEvents
            << " prediction(s) whose only explanation is one of those rows "
               "(neither hit nor false positive)\n";
        out << "  remaining      : " << clean.instance.capacity
            << " touches claimed, " << result.predictedEvents - clean.excludedEvents
            << " predictions scored\n";
        out << "  TP=" << clean.instance.truePositives
            << " FP=" << clean.instance.falsePositives
            << " FN=" << clean.instance.falseNegatives
            << "  precision=" << Percent(clean.instance.precision)
            << " recall=" << Percent(clean.instance.recall)
            << " f1=" << Percent(clean.instance.f1) << "\n";
        out << "  timing error vs the annotated instant (s): "
            << DescribePercentiles(clean.instance.signedDtSec, 3) << "\n";
        out << "  auxiliary: one-to-one event level\n";
        out << "  tol(s)     TP    FP    FN    precision  recall     f1\n";
        for (std::size_t i = 0; i < clean.tolerances.size(); ++i) {
            const ToleranceScore& score = clean.tolerances[i];
            out << "  " << std::setw(6) << FormatValue(score.toleranceSec, 2)
                << std::setw(6) << score.truePositives << std::setw(6)
                << score.falsePositives << std::setw(6) << score.falseNegatives
                << "    " << std::setw(9) << Percent(score.precision)
                << std::setw(9) << Percent(score.recall) << std::setw(9)
                << Percent(score.f1) << "\n";
        }
        out << "  auxiliary: count consistency  annotations="
            << clean.counts.records << "  exact=" << clean.counts.exactRecords
            << "  over=" << clean.counts.overRecords
            << "  under=" << clean.counts.underRecords << "  exact_rate="
            << Percent(clean.counts.records > 0
                           ? static_cast<double>(clean.counts.exactRecords) /
                                 static_cast<double>(clean.counts.records)
                           : 0.0)
            << "\n";
        out << "  flagged windows: usable ball " << clean.windowsWithUsableBall
            << "/" << clean.rows << ", usable feet " << clean.windowsWithUsableFeet
            << "/" << clean.rows << ", person height(px) "
            << DescribePercentiles(clean.personHeightPx, 1) << "\n";
        out << "                   ball->nearest foot distance(px) "
            << DescribePercentiles(clean.nearestUsableDistancePx) << "\n";
    }

    out << "\n-- false positives by source (adjacency "
        << FormatValue(context.options.adjacencySec, 2) << " s, duplicate "
        << FormatValue(context.options.duplicateSec, 2) << " s) --\n";
    const FalsePositiveBreakdown& breakdown = result.falsePositives;
    const int explained = breakdown.duplicates + breakdown.insideNoTouch;
    out << "  same touch, another frame : " << breakdown.duplicates << "\n";
    out << "  same moment, window full  : " << breakdown.inWindowNoSlot << "\n";
    out << "  inside a no_touch span    : " << breakdown.insideNoTouch << "\n";
    out << "  next to an annotation     : " << breakdown.adjacent << "\n";
    out << "  unannotated (detector)    : " << breakdown.unannotated << "\n";
    out << "  => the first three (" << explained + breakdown.inWindowNoSlot
        << ") are the same moment reported more than once or a moment the\n"
           "     annotator ruled out; " << breakdown.adjacent
        << " sit next to an annotation (the notation is coarser than the\n"
           "     detector); only " << breakdown.unannotated
        << " are detections nothing in the file supports\n";
    for (std::size_t i = 0; i < breakdown.examples.size(); ++i) {
        out << "        " << breakdown.examples[i] << "\n";
    }

    out << "\n-- auxiliary: one-to-one event level (tolerance sweep) --\n";
    out << "  tol(s)     TP    FP    FN    precision  recall     f1\n";
    for (std::size_t i = 0; i < result.tolerances.size(); ++i) {
        const ToleranceScore& score = result.tolerances[i];
        out << "  " << std::setw(6) << FormatValue(score.toleranceSec, 2)
            << std::setw(6) << score.truePositives << std::setw(6)
            << score.falsePositives << std::setw(6) << score.falseNegatives
            << "    " << std::setw(9) << Percent(score.precision) << std::setw(9)
            << Percent(score.recall) << std::setw(9) << Percent(score.f1) << "\n";
    }
    const double primaryTolerance = context.options.primaryToleranceSec;
    out << "\n-- headline (tolerance +-" << FormatValue(primaryTolerance, 2) << " s) --\n";
    out << "  TP=" << result.primary.truePositives
        << " FP=" << result.primary.falsePositives
        << " FN=" << result.primary.falseNegatives
        << "  precision=" << Percent(result.primary.precision)
        << " recall=" << Percent(result.primary.recall)
        << " f1=" << Percent(result.primary.f1) << "\n";
    out << "  timing error (prediction - annotation, s): "
        << DescribePercentiles(result.primary.signedDtSec, 3) << "\n";
    out << "  timing error magnitude (s)               : "
        << DescribePercentiles(result.primary.absDtSec, 3) << "\n";

    out << "\n-- auxiliary: count consistency (per annotation, tolerance +-"
        << FormatValue(primaryTolerance, 2) << " s) --\n";
    out << "  annotations=" << result.counts.records
        << "  exact=" << result.counts.exactRecords
        << "  over=" << result.counts.overRecords
        << "  under=" << result.counts.underRecords
        << "  exact_rate="
        << Percent(result.counts.records > 0
                       ? static_cast<double>(result.counts.exactRecords) /
                             static_cast<double>(result.counts.records)
                       : 0.0)
        << "\n";
    out << "  gt touches=" << result.counts.gtTouches
        << "  predicted in windows=" << result.counts.predictedInWindows
        << "  mean |count error|=" << FormatValue(result.counts.meanAbsCountError, 3)
        << "\n";
    out << "  mismatched annotations:\n";
    int printed = 0;
    for (std::size_t i = 0; i < result.counts.buckets.size(); ++i) {
        const CountBucket& bucket = result.counts.buckets[i];
        if (bucket.exact) continue;
        if (printed >= 20) {
            out << "        ... " << (result.counts.buckets.size() - i - 1)
                << " more\n";
            break;
        }
        out << "        gt#" << bucket.gtIndex << " @ "
            << FormatSeconds(bucket.startSec);
        if (bucket.endSec > bucket.startSec) {
            out << "-" << FormatSeconds(bucket.endSec);
        }
        out << "  gt=" << bucket.gtCount << " predicted=" << bucket.predictedCount
            << "\n";
        ++printed;
    }
    if (printed == 0) out << "        (none)\n";

    out << "\n-- diagnostics over the annotation windows --\n";
    const double records = result.diagnostics.records > 0
        ? static_cast<double>(result.diagnostics.records) : 1.0;
    out << "  windows with a ball detection      : "
        << Percent(result.diagnostics.withDetectedBall / records) << "\n";
    out << "  windows with an admissible ball    : "
        << Percent(result.diagnostics.withUsableBall / records) << "\n";
    out << "  windows with usable foot keypoints : "
        << Percent(result.diagnostics.withUsableFeet / records) << "\n";
    out << "  ball->nearest foot distance over annotated windows : "
        << DescribePercentiles(result.diagnostics.nearestUsableDistancePx) << "\n";
    out << "  ball->nearest foot distance over all frames        : "
        << DescribePercentiles(result.diagnostics.backgroundNearestUsableDistancePx)
        << "\n";
    out << "  person height in annotated windows (px): "
        << DescribePercentiles(result.diagnostics.personHeightPx, 1) << "\n";

    out << "\n-- misses and false positives --\n";
    if (result.failures.empty()) {
        out << "  (none)\n";
    } else {
        for (std::size_t i = 0; i < result.failures.size(); ++i) {
            AppendFailureGroup(out, result.failures[i]);
        }
    }

    out << "\n-- threshold calibration --\n";
    out << "  tau@720p = frame level enter threshold; a player's own height can\n"
           "  lower it further. The ratio sweep moves only that term.\n";
    out << "  primary = count aware; auxiliary = one-to-one at tolerance +-"
        << FormatValue(primaryTolerance, 2) << " s\n";
    out << "  ratio    tau@720p  | primary  P        R        F1     | auxiliary P        R        F1\n";
    for (std::size_t i = 0; i < result.sweep.size(); ++i) {
        const ThresholdSweepPoint& point = result.sweep[i];
        out << "  " << std::setw(7) << FormatValue(point.contactDistanceRatio, 4)
            << std::setw(10) << FormatValue(point.thresholdPxRef, 1)
            << "  |" << std::setw(9) << Percent(point.instancePrecision)
            << std::setw(9) << Percent(point.instanceRecall)
            << std::setw(9) << Percent(point.instanceF1)
            << "     |" << std::setw(9) << Percent(point.precision)
            << std::setw(9) << Percent(point.recall)
            << std::setw(9) << Percent(point.f1) << "\n";
    }

    out << "\n-- foot hint (only the annotations that carry one) --\n";
    out << "  records=" << result.footHintRecords
        << " correct=" << result.footHintCorrect << "\n";

    if (!context.groundTruth.warnings.empty()) {
        out << "\nground truth warnings: " << context.groundTruth.warnings.size()
            << "\n";
        const std::size_t limit = std::min<std::size_t>(
            context.groundTruth.warnings.size(), 10);
        for (std::size_t i = 0; i < limit; ++i) {
            out << "  " << context.groundTruth.warnings[i] << "\n";
        }
    }

    out << "\n" << FooterNote();
    return out.str();
}

bool WriteReportJson(const EvaluationResult& result,
                     const ReportContext& context, const std::string& path,
                     std::string* error) {
    Json::Value root;
    root["schema_version"] = 2;
    root["observations"] = context.observationsPath;
    root["ground_truth_source"] = context.groundTruthSource;
    root["ground_truth_from_json"] = context.groundTruthFromJson;
    root["ground_truth"]["source_kind"] = context.groundTruth.sourceKind;
    root["ground_truth"]["no_touch_spans"] =
        static_cast<Json::UInt>(context.groundTruth.noTouch.size());
    root["video"]["path"] = context.metadata.videoPath;
    root["video"]["camera_id"] = context.metadata.cameraId;
    root["video"]["fps"] = context.metadata.fps;
    root["video"]["width"] = context.metadata.width;
    root["video"]["height"] = context.metadata.height;
    root["video"]["stride"] = context.metadata.stride;
    root["frames"] = static_cast<Json::UInt64>(result.frames);
    root["duration_sec"] = result.durationSec;
    root["ground_truth"]["events"] = result.gtEvents;
    root["ground_truth"]["touches"] = result.gtTouches;
    root["ground_truth"]["covered_events"] = result.gtEventsCovered;
    root["covered"]["from_sec"] = result.coveredFromSec;
    root["covered"]["to_sec"] = result.coveredToSec;
    root["partial_run"] = result.partialRun;
    root["predicted_events"] = result.predictedEvents;
    root["params"] = DescribeTouchDetectorParams(context.options.params);
    root["primary_tolerance_sec"] = context.options.primaryToleranceSec;

    const auto fillScore = [](const ToleranceScore& score) {
        Json::Value value;
        value["tolerance_sec"] = score.toleranceSec;
        value["true_positives"] = score.truePositives;
        value["false_positives"] = score.falsePositives;
        value["false_negatives"] = score.falseNegatives;
        value["precision"] = score.precision;
        value["recall"] = score.recall;
        value["f1"] = score.f1;
        value["signed_dt_sec"]["n"] = score.signedDtSec.samples;
        value["signed_dt_sec"]["mean"] = score.signedDtSec.mean;
        value["signed_dt_sec"]["p50"] = score.signedDtSec.p50;
        value["signed_dt_sec"]["p95"] = score.signedDtSec.p95;
        value["signed_dt_sec"]["max"] = score.signedDtSec.max;
        value["abs_dt_sec"]["n"] = score.absDtSec.samples;
        value["abs_dt_sec"]["mean"] = score.absDtSec.mean;
        value["abs_dt_sec"]["p50"] = score.absDtSec.p50;
        value["abs_dt_sec"]["p95"] = score.absDtSec.p95;
        value["abs_dt_sec"]["max"] = score.absDtSec.max;
        return value;
    };

    // Primary metric: windows plus capacity, i.e. what the annotation promises.
    // Shared by both views, so the full one and the one without the unreadable
    // rows are always described the same way.
    const auto fillInstance = [&context](const InstanceScore& score) {
        Json::Value value;
        value["point_window_sec"] = score.pointWindowSec;
        value["window_pad_sec"] = score.windowPadSec;
        value["adjacency_sec"] = context.options.adjacencySec;
        value["duplicate_sec"] = context.options.duplicateSec;
        value["capacity"] = score.capacity;
        value["true_positives"] = score.truePositives;
        value["false_positives"] = score.falsePositives;
        value["false_negatives"] = score.falseNegatives;
        value["precision"] = score.precision;
        value["recall"] = score.recall;
        value["f1"] = score.f1;
        value["windows_with_event"] = score.windowsWithEvent;
        value["windows_exact"] = score.windowsExact;
        value["suppressed_by_refractory"] = score.suppressedByRefractory;
        value["rejected_by_speed_gate"] = score.rejectedBySpeedGate;
        value["rejected_by_ball_gate"] = score.rejectedByBallGate;
        value["signed_dt_sec"]["n"] = score.signedDtSec.samples;
        value["signed_dt_sec"]["mean"] = score.signedDtSec.mean;
        value["signed_dt_sec"]["p50"] = score.signedDtSec.p50;
        value["signed_dt_sec"]["p95"] = score.signedDtSec.p95;
        value["signed_dt_sec"]["max"] = score.signedDtSec.max;
        value["abs_dt_sec"]["n"] = score.absDtSec.samples;
        value["abs_dt_sec"]["mean"] = score.absDtSec.mean;
        value["abs_dt_sec"]["p50"] = score.absDtSec.p50;
        value["abs_dt_sec"]["p95"] = score.absDtSec.p95;
        value["abs_dt_sec"]["max"] = score.absDtSec.max;
        return value;
    };
    root["primary_count_aware"] = fillInstance(result.instance);

    root["false_positive_breakdown"]["duplicates"] =
        result.falsePositives.duplicates;
    root["false_positive_breakdown"]["inside_no_touch"] =
        result.falsePositives.insideNoTouch;
    root["false_positive_breakdown"]["in_window_no_slot"] =
        result.falsePositives.inWindowNoSlot;
    root["false_positive_breakdown"]["adjacent"] =
        result.falsePositives.adjacent;
    root["false_positive_breakdown"]["unannotated"] =
        result.falsePositives.unannotated;
    root["false_positive_breakdown"]["examples"] = Json::arrayValue;
    for (std::size_t i = 0; i < result.falsePositives.examples.size(); ++i) {
        root["false_positive_breakdown"]["examples"].append(
            result.falsePositives.examples[i]);
    }

    root["tolerance_sweep"] = Json::arrayValue;
    for (std::size_t i = 0; i < result.tolerances.size(); ++i) {
        root["tolerance_sweep"].append(fillScore(result.tolerances[i]));
    }
    root["primary"] = fillScore(result.primary);

    // Second view: the flagged rows are gone, and with them the predictions only
    // those rows could explain.
    root["low_quality"]["rows"] = result.lowQuality.rows;
    root["low_quality"]["touches"] = result.lowQuality.touches;
    root["low_quality"]["excluded_events"] = result.lowQuality.excludedEvents;
    root["low_quality"]["scored_events"] =
        result.predictedEvents - result.lowQuality.excludedEvents;
    root["low_quality"]["primary_count_aware"] =
        fillInstance(result.lowQuality.instance);
    root["low_quality"]["tolerance_sweep"] = Json::arrayValue;
    for (std::size_t i = 0; i < result.lowQuality.tolerances.size(); ++i) {
        root["low_quality"]["tolerance_sweep"].append(
            fillScore(result.lowQuality.tolerances[i]));
    }
    root["low_quality"]["primary"] = fillScore(result.lowQuality.primary);
    root["low_quality"]["counts"]["records"] = result.lowQuality.counts.records;
    root["low_quality"]["counts"]["exact_records"] =
        result.lowQuality.counts.exactRecords;
    root["low_quality"]["counts"]["over_records"] =
        result.lowQuality.counts.overRecords;
    root["low_quality"]["counts"]["under_records"] =
        result.lowQuality.counts.underRecords;
    root["low_quality"]["counts"]["gt_touches"] =
        result.lowQuality.counts.gtTouches;
    root["low_quality"]["counts"]["predicted_in_windows"] =
        result.lowQuality.counts.predictedInWindows;
    root["low_quality"]["counts"]["mean_abs_count_error"] =
        result.lowQuality.counts.meanAbsCountError;
    root["low_quality"]["windows"]["with_usable_ball"] =
        result.lowQuality.windowsWithUsableBall;
    root["low_quality"]["windows"]["with_usable_feet"] =
        result.lowQuality.windowsWithUsableFeet;

    root["counts"]["records"] = result.counts.records;
    root["counts"]["exact_records"] = result.counts.exactRecords;
    root["counts"]["over_records"] = result.counts.overRecords;
    root["counts"]["under_records"] = result.counts.underRecords;
    root["counts"]["gt_touches"] = result.counts.gtTouches;
    root["counts"]["predicted_in_windows"] = result.counts.predictedInWindows;
    root["counts"]["mean_abs_count_error"] = result.counts.meanAbsCountError;
    root["counts"]["buckets"] = Json::arrayValue;
    for (std::size_t i = 0; i < result.counts.buckets.size(); ++i) {
        const CountBucket& bucket = result.counts.buckets[i];
        Json::Value item;
        item["gt_index"] = bucket.gtIndex;
        item["start_sec"] = bucket.startSec;
        item["end_sec"] = bucket.endSec;
        item["gt_count"] = bucket.gtCount;
        item["predicted_count"] = bucket.predictedCount;
        item["exact"] = bucket.exact;
        root["counts"]["buckets"].append(item);
    }

    root["diagnostics"]["records"] = result.diagnostics.records;
    root["diagnostics"]["windows_with_detected_ball"] =
        result.diagnostics.withDetectedBall;
    root["diagnostics"]["windows_with_usable_ball"] =
        result.diagnostics.withUsableBall;
    root["diagnostics"]["windows_with_usable_feet"] =
        result.diagnostics.withUsableFeet;
    const auto fillPercentiles = [](const Percentiles& value) {
        Json::Value json;
        json["n"] = value.samples;
        json["mean"] = value.mean;
        json["p50"] = value.p50;
        json["p90"] = value.p90;
        json["p95"] = value.p95;
        json["p99"] = value.p99;
        json["min"] = value.min;
        json["max"] = value.max;
        return json;
    };
    root["diagnostics"]["nearest_distance_px"] =
        fillPercentiles(result.diagnostics.nearestDistancePx);
    root["diagnostics"]["nearest_usable_distance_px"] =
        fillPercentiles(result.diagnostics.nearestUsableDistancePx);
    root["diagnostics"]["background_nearest_usable_distance_px"] =
        fillPercentiles(result.diagnostics.backgroundNearestUsableDistancePx);
    root["diagnostics"]["person_height_px"] =
        fillPercentiles(result.diagnostics.personHeightPx);
    root["low_quality"]["windows"]["person_height_px"] =
        fillPercentiles(result.lowQuality.personHeightPx);
    root["low_quality"]["windows"]["nearest_usable_distance_px"] =
        fillPercentiles(result.lowQuality.nearestUsableDistancePx);

    root["failures"] = Json::arrayValue;
    for (std::size_t i = 0; i < result.failures.size(); ++i) {
        const FailureGroup& group = result.failures[i];
        Json::Value item;
        item["reason"] = group.reason;
        item["count"] = group.count;
        item["examples"] = Json::arrayValue;
        for (std::size_t e = 0; e < group.examples.size(); ++e) {
            item["examples"].append(group.examples[e]);
        }
        root["failures"].append(item);
    }

    root["calibration"] = Json::arrayValue;
    for (std::size_t i = 0; i < result.sweep.size(); ++i) {
        const ThresholdSweepPoint& point = result.sweep[i];
        Json::Value item;
        item["contact_distance_ratio"] = point.contactDistanceRatio;
        item["threshold_px_ref"] = point.thresholdPxRef;
        item["tolerance_sec"] = point.toleranceSec;
        item["true_positives"] = point.truePositives;
        item["false_positives"] = point.falsePositives;
        item["false_negatives"] = point.falseNegatives;
        item["precision"] = point.precision;
        item["recall"] = point.recall;
        item["f1"] = point.f1;
        item["instance_precision"] = point.instancePrecision;
        item["instance_recall"] = point.instanceRecall;
        item["instance_f1"] = point.instanceF1;
        root["calibration"].append(item);
    }

    root["events"] = Json::arrayValue;
    for (std::size_t i = 0; i < result.events.size(); ++i) {
        const TouchEvent& event = result.events[i];
        Json::Value item;
        item["frame_index"] = static_cast<Json::UInt64>(event.frameIndex);
        item["first_frame_index"] =
            static_cast<Json::UInt64>(event.firstFrameIndex);
        item["timestamp_sec"] = event.timestampSec;
        item["person_track_id"] = event.personTrackId;
        item["foot"] = event.leftFoot ? "left" : "right";
        item["distance_px"] = event.distancePx;
        item["threshold_px"] = event.thresholdPx;
        item["ball_confidence"] = event.ballConfidence;
        item["ball_tracked"] = event.ballTracked;
        item["ball_track_id"] = event.ballTrackId;
        item["contact_frames"] = event.contactFrames;
        item["speed_before_px_per_sec"] = event.speedBeforePxPerSec;
        item["speed_after_px_per_sec"] = event.speedAfterPxPerSec;
        item["gt_index"] = result.eventGtIndex.size() > i
            ? result.eventGtIndex[i] : -1;
        item["verdict"] = result.falsePositives.eventVerdict.size() > i
            ? EventVerdictName(result.falsePositives.eventVerdict[i])
            : "assigned";
        root["events"].append(item);
    }

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    builder["emitUTF8"] = true;
    std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        if (error) *error = "cannot write report '" + path + "'";
        return false;
    }
    file << Json::writeString(builder, root) << '\n';
    return true;
}

} // namespace touch_eval

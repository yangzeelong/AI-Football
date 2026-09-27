#include <touch_eval/TouchDetector.hpp>
#include <touch_eval/TouchGroundTruth.hpp>
#include <touch_eval/TouchMetrics.hpp>

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using touch_eval::ContactLevel;
using touch_eval::FrameEvidence;
using touch_eval::GroundTruth;
using touch_eval::GroundTruthEvent;
using touch_eval::TouchDetector;
using touch_eval::TouchDetectorParams;
using touch_eval::TouchEvent;
using touch_eval::TouchFrameInput;

TouchFrameInput MakeFrame(uint64_t index, double timestampSec) {
    TouchFrameInput frame;
    frame.frameIndex = index;
    frame.timestampSec = timestampSec;
    frame.frameWidth = 1280;
    frame.frameHeight = 720;
    return frame;
}

/// Person whose only valid keypoint is the left ankle (foot slot 0). Every
/// slot carries a confident score, so a test only has to mark one as valid.
TouchFrameInput::Person MakePerson(int trackId, float heightPx, float footX,
                                   float footY) {
    TouchFrameInput::Person person;
    person.trackId = trackId;
    person.heightPx = heightPx;
    for (int slot = 0; slot < touch_eval::kFootKeypointCount; ++slot) {
        person.feet[slot].confidence = 0.9f;
    }
    person.feet[0].x = footX;
    person.feet[0].y = footY;
    person.feet[0].valid = true;
    return person;
}

TouchFrameInput::Ball MakeBall(float x, float y, float confidence,
                               int trackId) {
    TouchFrameInput::Ball ball;
    ball.x = x;
    ball.y = y;
    ball.confidence = confidence;
    ball.trackId = trackId;
    ball.observed = true;
    return ball;
}

/// Frame with one player (left ankle at x=100, y=100) and the ball on that
/// line, so the ball distance is exactly `distancePx`.
TouchFrameInput DistanceFrame(uint64_t index, float distancePx,
                             float personHeightPx = 600.0f) {
    TouchFrameInput frame = MakeFrame(index, static_cast<double>(index) / 60.0);
    frame.persons.push_back(MakePerson(1, personHeightPx, 100.0f, 100.0f));
    frame.balls.push_back(MakeBall(100.0f + distancePx, 100.0f, 0.9f, 3));
    return frame;
}

/// The same frame at an arbitrary time, ready to be aggregated.
FrameEvidence EvidenceAt(double timestampSec, float distancePx,
                         const TouchDetectorParams& params,
                         float personHeightPx = 600.0f,
                         int personTrackId = 1) {
    TouchFrameInput frame = MakeFrame(
        static_cast<uint64_t>(std::llround(timestampSec * 60.0)), timestampSec);
    frame.persons.push_back(
        MakePerson(personTrackId, personHeightPx, 100.0f, 100.0f));
    frame.balls.push_back(MakeBall(100.0f + distancePx, 100.0f, 0.9f, 3));
    return touch_eval::BuildFrameEvidence(frame, params);
}

/// Two frames of contact at `timestampSec`, then four frames far away so the
/// cluster closes before the next touch.
void AppendTouch(std::vector<FrameEvidence>* frames, double timestampSec,
                 float distancePx, const TouchDetectorParams& params,
                 float personHeightPx = 600.0f, int personTrackId = 1) {
    frames->push_back(
        EvidenceAt(timestampSec, distancePx, params, personHeightPx, personTrackId));
    frames->push_back(EvidenceAt(timestampSec + 1.0 / 60.0, distancePx, params,
                                 personHeightPx, personTrackId));
    for (int i = 1; i <= 4; ++i) {
        frames->push_back(EvidenceAt(timestampSec + i * 1.0 / 60.0, 500.0f, params,
                                     personHeightPx, personTrackId));
    }
}

/// Frame with the ball at an explicit position: used to script a speed profile.
FrameEvidence BallAt(double timestampSec, float ballX, float ballY,
                     const TouchDetectorParams& params) {
    TouchFrameInput frame = MakeFrame(
        static_cast<uint64_t>(std::llround(timestampSec * 60.0)), timestampSec);
    frame.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));
    frame.balls.push_back(MakeBall(ballX, ballY, 0.9f, 3));
    return touch_eval::BuildFrameEvidence(frame, params);
}

/// Frame with no ball at all: the ball left the frame, so the aggregator gets
/// no speed sample from it.
FrameEvidence NoBallAt(double timestampSec, const TouchDetectorParams& params) {
    TouchFrameInput frame = MakeFrame(
        static_cast<uint64_t>(std::llround(timestampSec * 60.0)), timestampSec);
    frame.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));
    return touch_eval::BuildFrameEvidence(frame, params);
}

/// The ball closes in on the left ankle at `stepPx` per frame, stops on it for
/// one frame and is not reported afterwards.
std::vector<FrameEvidence> ApproachScenario(float stepPx,
                                            const TouchDetectorParams& params) {
    std::vector<FrameEvidence> frames;
    frames.push_back(BallAt(0.0, 100.0f + 2.0f * stepPx, 100.0f, params));
    frames.push_back(BallAt(1.0 / 60.0, 100.0f + stepPx, 100.0f, params));
    frames.push_back(BallAt(2.0 / 60.0, 110.0f, 100.0f, params));
    frames.push_back(BallAt(3.0 / 60.0, 110.0f, 100.0f, params));
    for (int i = 4; i <= 7; ++i) {
        frames.push_back(NoBallAt(i / 60.0, params));
    }
    return frames;
}

/// Two contacts of the same person and foot, `gapFrames` apart, each closed by
/// frames that are far from the foot.
std::vector<FrameEvidence> TwoContactsScenario(int gapFrames,
                                               const TouchDetectorParams& params) {
    std::vector<FrameEvidence> frames;
    const int starts[2] = {0, gapFrames};
    for (int c = 0; c < 2; ++c) {
        const int start = starts[c];
        frames.push_back(EvidenceAt(start / 60.0, 25.0f, params));
        frames.push_back(EvidenceAt((start + 1) / 60.0, 25.0f, params));
        for (int i = 2; i <= 5; ++i) {
            frames.push_back(EvidenceAt((start + i) / 60.0, 500.0f, params));
        }
    }
    return frames;
}

} // namespace

TEST(TouchDetectorTest, ContactThresholdScalesWithFrameAndPerson) {
    TouchDetectorParams params;

    // Frame term: 720 * 0.045 = 32.4 px. A 200 px tall player reaches
    // 200 * 0.12 = 24 px, which is the binding term.
    EXPECT_NEAR(touch_eval::ContactThresholdPx(params, 1280, 720, 200.0f),
                24.0f, 1e-4f);
    // 600 px tall: 72 px, clamped to the 64 px ceiling, so the frame term of
    // 32.4 px binds.
    EXPECT_NEAR(touch_eval::ContactThresholdPx(params, 1280, 720, 600.0f),
                32.4f, 1e-4f);
    // Missing metadata must not disable the floor.
    EXPECT_NEAR(touch_eval::ContactThresholdPx(params, 0, 0, 200.0f), 20.0f,
                1e-4f);
}

TEST(TouchDetectorTest, CandidatesCoverBothSidesAndClosestKeypoint) {
    TouchFrameInput frame = MakeFrame(0, 0.0);
    TouchFrameInput::Person person = MakePerson(7, 600.0f, 100.0f, 100.0f);
    // Left heel is as close as the ankle, right big toe is far away.
    person.feet[4].x = 110.0f;
    person.feet[4].y = 100.0f;
    person.feet[4].valid = true;
    person.feet[5].x = 300.0f;
    person.feet[5].y = 300.0f;
    person.feet[5].valid = true;
    frame.persons.push_back(person);
    frame.balls.push_back(MakeBall(105.0f, 100.0f, 0.9f, 3));

    TouchDetectorParams params;
    const std::vector<touch_eval::ContactCandidate> candidates =
        touch_eval::CollectContactCandidates(frame, params, 6);
    ASSERT_EQ(candidates.size(), 2U);
    EXPECT_TRUE(candidates[0].leftFoot);
    EXPECT_EQ(candidates[0].footSlot, 0); // ankle wins the tie with the heel
    EXPECT_NEAR(candidates[0].distancePx, 5.0f, 1e-3f);
    EXPECT_TRUE(candidates[0].ballTracked);
    EXPECT_FALSE(candidates[1].leftFoot);
    EXPECT_NEAR(candidates[1].distancePx, std::hypot(195.0f, 200.0f), 1e-3f);

    const touch_eval::FrameContact contact =
        touch_eval::SelectContact(frame, candidates, params);
    EXPECT_TRUE(contact.hasBall);
    EXPECT_EQ(contact.level, ContactLevel::Contact); // 5 px <= 32.4 px
}

TEST(TouchDetectorTest, PredictedAndLowConfidenceBallsAreNotEvidence) {
    TouchFrameInput frame = MakeFrame(0, 0.0);
    frame.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));

    TouchFrameInput::Ball predicted = MakeBall(101.0f, 100.0f, 0.9f, -1);
    predicted.observed = false;
    frame.balls.push_back(predicted);
    frame.balls.push_back(MakeBall(101.0f, 100.0f, 0.01f, -1));

    TouchDetectorParams params;
    EXPECT_TRUE(touch_eval::CollectContactCandidates(frame, params, 6).empty());

    const touch_eval::FrameContact contact =
        touch_eval::SelectContact(frame, {}, params);
    EXPECT_FALSE(contact.hasBall);
    EXPECT_EQ(contact.level, ContactLevel::None);
    EXPECT_EQ(contact.observedBallCount, 0);
    EXPECT_EQ(contact.personsWithFeet, 1);

    TouchDetector detector(params);
    EXPECT_TRUE(detector.Push(frame).empty());
    EXPECT_TRUE(detector.Flush().empty());
}

TEST(TouchDetectorTest, TrackedBallWinsOverACloserUntrackedCandidate) {
    TouchFrameInput frame = MakeFrame(0, 0.0);
    frame.persons.push_back(MakePerson(2, 600.0f, 100.0f, 100.0f));
    frame.balls.push_back(MakeBall(105.0f, 100.0f, 0.9f, -1)); // 5 px
    frame.balls.push_back(MakeBall(130.0f, 100.0f, 0.9f, 4));  // 30 px, tracked

    TouchDetectorParams params;
    const std::vector<touch_eval::ContactCandidate> candidates =
        touch_eval::CollectContactCandidates(frame, params, 6);
    // Only the left side has a valid keypoint, so one candidate per ball.
    ASSERT_EQ(candidates.size(), 2U);
    EXPECT_FALSE(candidates[0].ballTracked);
    EXPECT_TRUE(candidates[1].ballTracked);

    const touch_eval::FrameContact contact =
        touch_eval::SelectContact(frame, candidates, params);
    EXPECT_TRUE(contact.candidate.ballTracked);
    EXPECT_EQ(contact.candidate.ballTrackId, 4);
    EXPECT_NEAR(contact.candidate.distancePx, 30.0f, 1e-3f);
    EXPECT_EQ(contact.level, ContactLevel::Contact);

    params.preferTrackedBall = false;
    const touch_eval::FrameContact nearest =
        touch_eval::SelectContact(frame, candidates, params);
    EXPECT_FALSE(nearest.candidate.ballTracked);
    EXPECT_NEAR(nearest.candidate.distancePx, 5.0f, 1e-3f);
}

TEST(TouchDetectorTest, EmitsOneEventForOneTouch) {
    const TouchDetectorParams params;
    TouchDetector detector(params);
    // Enter threshold is 32.4 px, exit is 45.36 px.
    const float distances[] = {60.0f, 40.0f, 20.0f, 8.0f, 20.0f, 40.0f, 60.0f};

    std::vector<TouchEvent> events;
    for (uint64_t i = 0; i < 7; ++i) {
        const std::vector<TouchEvent> emitted =
            detector.Push(DistanceFrame(i, distances[i]));
        events.insert(events.end(), emitted.begin(), emitted.end());
    }
    const std::vector<TouchEvent> flushed = detector.Flush();
    events.insert(events.end(), flushed.begin(), flushed.end());

    ASSERT_EQ(events.size(), 1U);
    const TouchEvent& event = events[0];
    EXPECT_EQ(event.frameIndex, 3U); // closest approach
    EXPECT_EQ(event.firstFrameIndex, 2U);
    EXPECT_NEAR(event.distancePx, 8.0f, 1e-3f);
    EXPECT_EQ(event.personTrackId, 1);
    EXPECT_TRUE(event.leftFoot);
    EXPECT_TRUE(event.ballTracked);
    EXPECT_EQ(event.ballTrackId, 3);
    EXPECT_EQ(event.contactFrames, 3);
}

TEST(TouchDetectorTest, RejectsASingleFrameBlip) {
    const TouchDetectorParams params;
    TouchDetector detector(params);
    std::vector<TouchEvent> events = detector.Push(DistanceFrame(0, 20.0f));
    for (uint64_t i = 1; i < 10; ++i) {
        const std::vector<TouchEvent> emitted =
            detector.Push(DistanceFrame(i, 200.0f));
        events.insert(events.end(), emitted.begin(), emitted.end());
    }
    const std::vector<TouchEvent> flushed = detector.Flush();
    events.insert(events.end(), flushed.begin(), flushed.end());
    EXPECT_TRUE(events.empty());
}

TEST(TouchDetectorTest, ToleratesFewFramesOfLostBall) {
    const TouchDetectorParams params;
    TouchDetector detector(params);
    std::vector<TouchEvent> events;
    const float distances[] = {20.0f, 19.0f, 21.0f};
    for (uint64_t i = 0; i < 3; ++i) {
        const std::vector<TouchEvent> emitted =
            detector.Push(DistanceFrame(i, distances[i]));
        events.insert(events.end(), emitted.begin(), emitted.end());
    }
    // The ball disappears for three frames (occluded by the foot), then the
    // touch continues.
    for (uint64_t i = 3; i < 6; ++i) {
        TouchFrameInput frame = MakeFrame(i, static_cast<double>(i) / 60.0);
        frame.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));
        const std::vector<TouchEvent> emitted = detector.Push(frame);
        events.insert(events.end(), emitted.begin(), emitted.end());
    }
    for (uint64_t i = 6; i < 8; ++i) {
        const std::vector<TouchEvent> emitted =
            detector.Push(DistanceFrame(i, 22.0f));
        events.insert(events.end(), emitted.begin(), emitted.end());
    }
    const std::vector<TouchEvent> flushed = detector.Flush();
    events.insert(events.end(), flushed.begin(), flushed.end());

    ASSERT_EQ(events.size(), 1U);
    EXPECT_EQ(events[0].frameIndex, 1U);
    EXPECT_EQ(events[0].firstFrameIndex, 0U);
    EXPECT_EQ(events[0].contactFrames, 5);
}

TEST(TouchDetectorTest, SplitsTwoTouchesSeparatedByTheExitThreshold) {
    TouchDetectorParams params; // 32.4 px enter, 45.36 px exit
    // This test is about the split rule inside one cluster; the refractory
    // window (a cross-cluster rule) would merge the two events again.
    params.refractoryFrames = 0;
    TouchDetector detector(params);

    // Two close encounters: the ball leaves the exit threshold for two frames
    // (60 px) and the whole excursion lasts 16 frames, so the cluster holds
    // two separate touches.
    std::vector<float> distances;
    for (int i = 0; i < 4; ++i) distances.push_back(20.0f);
    for (int i = 0; i < 5; ++i) distances.push_back(40.0f);
    for (int i = 0; i < 2; ++i) distances.push_back(60.0f);
    for (int i = 0; i < 5; ++i) distances.push_back(40.0f);
    for (int i = 0; i < 4; ++i) distances.push_back(18.0f);

    std::vector<TouchEvent> events;
    for (std::size_t i = 0; i < distances.size(); ++i) {
        const std::vector<TouchEvent> emitted =
            detector.Push(DistanceFrame(i, distances[i]));
        events.insert(events.end(), emitted.begin(), emitted.end());
    }
    const std::vector<TouchEvent> flushed = detector.Flush();
    events.insert(events.end(), flushed.begin(), flushed.end());

    ASSERT_EQ(events.size(), 2U);
    EXPECT_EQ(events[0].frameIndex, 0U);
    EXPECT_NEAR(events[0].distancePx, 20.0f, 1e-3f);
    EXPECT_EQ(events[0].contactFrames, 4);
    EXPECT_EQ(events[1].frameIndex, 16U);
    EXPECT_NEAR(events[1].distancePx, 18.0f, 1e-3f);
    EXPECT_EQ(events[1].contactFrames, 4);
}

TEST(TouchDetectorTest, KeepsOneEventWhenTheBallOnlyRisesInsideTheBand) {
    TouchDetectorParams params; // exit threshold is 45.36 px
    TouchDetector detector(params);

    // 40 px never leaves the hysteresis band, so this is one long touch even
    // though the distance rises between two contact runs.
    std::vector<float> distances;
    for (int i = 0; i < 4; ++i) distances.push_back(20.0f);
    for (int i = 0; i < 16; ++i) distances.push_back(40.0f);
    for (int i = 0; i < 4; ++i) distances.push_back(20.0f);

    std::vector<TouchEvent> events;
    for (std::size_t i = 0; i < distances.size(); ++i) {
        const std::vector<TouchEvent> emitted =
            detector.Push(DistanceFrame(i, distances[i]));
        events.insert(events.end(), emitted.begin(), emitted.end());
    }
    const std::vector<TouchEvent> flushed = detector.Flush();
    events.insert(events.end(), flushed.begin(), flushed.end());

    ASSERT_EQ(events.size(), 1U);
    EXPECT_EQ(events[0].contactFrames, 8);
}

TEST(TouchGroundTruthTest, ParsesEveryNotationUsedInTheLabel) {
    const std::string text =
        "0.01S  (1次)\n"
        "0.07S（2次）\n"
        "00:29S  (1次)\n"
        "00:34S - 00:35 S (1次)\n"
        "00:38S - 0.39 (1次)\n"
        "01:46S - 01:47S (2次) - 左右脚各1次\n"
        "04:22S (1次)\n";

    const GroundTruth groundTruth = touch_eval::ParseTouchLabelText(text, "inline");
    ASSERT_EQ(groundTruth.events.size(), 7U);
    ASSERT_TRUE(groundTruth.warnings.empty());

    // Dotted "0.01" is the mm:ss typo used in the first lines of the file.
    EXPECT_NEAR(groundTruth.events[0].startSec, 1.0, 1e-9);
    EXPECT_FALSE(groundTruth.events[0].IsInterval());
    EXPECT_EQ(groundTruth.events[0].warnings.size(), 1U);
    EXPECT_NEAR(groundTruth.events[1].startSec, 7.0, 1e-9);
    EXPECT_EQ(groundTruth.events[1].count, 2);
    EXPECT_TRUE(groundTruth.events[2].warnings.empty());

    EXPECT_NEAR(groundTruth.events[2].startSec, 29.0, 1e-9);
    EXPECT_FALSE(groundTruth.events[2].IsInterval());

    // Interval annotation: the touch process lasted more than one second.
    EXPECT_NEAR(groundTruth.events[3].startSec, 34.0, 1e-9);
    EXPECT_NEAR(groundTruth.events[3].endSec, 35.0, 1e-9);
    EXPECT_TRUE(groundTruth.events[3].IsInterval());

    // "0.39" is the typo for 00:39.
    EXPECT_NEAR(groundTruth.events[4].startSec, 38.0, 1e-9);
    EXPECT_NEAR(groundTruth.events[4].endSec, 39.0, 1e-9);

    EXPECT_EQ(groundTruth.events[5].count, 2);
    EXPECT_EQ(groundTruth.events[5].footHint, "both");
    EXPECT_NEAR(groundTruth.events[5].startSec, 106.0, 1e-9);
    EXPECT_NEAR(groundTruth.events[5].endSec, 107.0, 1e-9);
    EXPECT_EQ(groundTruth.events[5].note, "左右脚各1次");

    EXPECT_NEAR(groundTruth.events[6].startSec, 262.0, 1e-9);
    EXPECT_EQ(groundTruth.TouchCount(), 9);
}

TEST(TouchGroundTruthTest, SkipsLinesWithoutTimestamp) {
    const GroundTruth groundTruth = touch_eval::ParseTouchLabelText(
        "触球记录\n\n# comment\n00:10S\n", "inline");
    ASSERT_EQ(groundTruth.events.size(), 1U);
    EXPECT_EQ(groundTruth.warnings.size(), 1U);
    EXPECT_NEAR(groundTruth.events[0].startSec, 10.0, 1e-9);
    // No "(N次)" on the line: the parser assumes a single touch and says so.
    EXPECT_EQ(groundTruth.events[0].count, 1);
    EXPECT_EQ(groundTruth.events[0].warnings.size(), 1U);
}

TEST(TouchGroundTruthTest, JsonRoundTripPreservesEventsAndWarnings) {
    GroundTruth groundTruth = touch_eval::ParseTouchLabelText(
        "00:29S  (1次)\n00:34S - 00:35 S (1次)\n", "inline");
    const std::string path = "/tmp/nexusflow_touch_eval_gt_test.json";

    std::string error;
    ASSERT_TRUE(touch_eval::WriteGroundTruthJson(groundTruth, path, &error))
        << error;

    GroundTruth reloaded;
    ASSERT_TRUE(touch_eval::ReadGroundTruthJson(path, &reloaded, &error))
        << error;
    std::remove(path.c_str());

    ASSERT_EQ(reloaded.events.size(), 2U);
    EXPECT_EQ(reloaded.source, "inline");
    EXPECT_NEAR(reloaded.events[0].startSec, 29.0, 1e-9);
    EXPECT_NEAR(reloaded.events[1].endSec, 35.0, 1e-9);
    EXPECT_EQ(reloaded.events[1].rawLine, "00:34S - 00:35 S (1次)");
    EXPECT_EQ(reloaded.TouchCount(), 2);
}

TEST(TouchGroundTruthTest, ParsesTouchesCsvWithNegativeSpans) {
    const std::string text =
        "# touch annotations v2\n"
        "index,type,t_start_sec,t_end_sec,count,foot,player,note\n"
        "1,touch,1.00,1.00,1,,,\n"
        "2,touch,34.20,35.10,1,right,P7,\"long touch, held\"\n"
        "3,no_touch,12.00,15.00,0,,,keeper picks it up\n"
        "4,touch,2:10.50,2:10.50,1,left,,\n";

    const GroundTruth groundTruth = touch_eval::ParseTouchCsvText(text, "inline");
    ASSERT_TRUE(groundTruth.warnings.empty());
    EXPECT_EQ(groundTruth.sourceKind, "csv");
    // A 0.01 s CSV instant stands for about half a second, not a whole one.
    EXPECT_NEAR(groundTruth.defaultPointWindowSec, 0.5, 1e-9);

    ASSERT_EQ(groundTruth.events.size(), 3U);
    ASSERT_EQ(groundTruth.noTouch.size(), 1U);
    EXPECT_NEAR(groundTruth.events[0].startSec, 1.0, 1e-9);
    EXPECT_FALSE(groundTruth.events[0].IsInterval());
    EXPECT_NEAR(groundTruth.events[1].startSec, 34.2, 1e-9);
    EXPECT_NEAR(groundTruth.events[1].endSec, 35.1, 1e-9);
    EXPECT_EQ(groundTruth.events[1].footHint, "right");
    EXPECT_EQ(groundTruth.events[1].playerHint, "P7");
    EXPECT_EQ(groundTruth.events[1].note, "long touch, held");
    EXPECT_NEAR(groundTruth.events[2].startSec, 130.5, 1e-9);

    EXPECT_NEAR(groundTruth.noTouch[0].startSec, 12.0, 1e-9);
    EXPECT_NEAR(groundTruth.noTouch[0].endSec, 15.0, 1e-9);
    EXPECT_EQ(groundTruth.noTouch[0].count, 0);
    EXPECT_EQ(groundTruth.TouchCount(), 3);
}

TEST(TouchGroundTruthTest, FrameColumnsWinOverSeconds) {
    const std::string text =
        "# fps: 60\n"
        "index,type,t_start_sec,t_end_sec,t_start_frame,t_end_frame,count,foot,player,note\n"
        "1,touch,999.00,999.00,16010,16010,1,right,,pinned by frame\n"
        "2,touch,2.00,2.00,,,1,,,\n"
        "3,no_touch,0.00,0.00,200,250,0,,,negative span by frame\n";

    const GroundTruth groundTruth =
        touch_eval::ParseTouchCsvText(text, "inline");
    EXPECT_NEAR(groundTruth.fps, 60.0, 1e-9);
    ASSERT_TRUE(groundTruth.warnings.empty());
    ASSERT_EQ(groundTruth.events.size(), 2U);
    ASSERT_EQ(groundTruth.noTouch.size(), 1U);

    // A stated frame beats the seconds next to it.
    EXPECT_NEAR(groundTruth.events[0].startSec, 16010.0 / 60.0, 1e-9);
    EXPECT_EQ(groundTruth.events[0].startFrame, 16010);
    EXPECT_EQ(groundTruth.events[0].endFrame, 16010);
    EXPECT_FALSE(groundTruth.events[0].IsInterval());

    // A seconds-only row gets its frames derived from the declared rate.
    EXPECT_NEAR(groundTruth.events[1].startSec, 2.0, 1e-9);
    EXPECT_EQ(groundTruth.events[1].startFrame, 120);
    EXPECT_EQ(groundTruth.events[1].endFrame, 120);

    // A frame range becomes an interval, again overriding the seconds.
    EXPECT_NEAR(groundTruth.noTouch[0].startSec, 200.0 / 60.0, 1e-9);
    EXPECT_NEAR(groundTruth.noTouch[0].endSec, 250.0 / 60.0, 1e-9);
    EXPECT_TRUE(groundTruth.noTouch[0].IsInterval());
}

TEST(TouchGroundTruthTest, FrameWithoutFpsIsReportedNotGuessed) {
    const std::string text =
        "index,type,t_start_sec,t_end_sec,frame,count,foot,player,note\n"
        "1,touch,5.00,5.00,300,1,,,\n";

    // No fps anywhere: keep the seconds and say so rather than inventing a rate.
    const GroundTruth withoutFps =
        touch_eval::ParseTouchCsvText(text, "inline");
    ASSERT_EQ(withoutFps.events.size(), 1U);
    EXPECT_NEAR(withoutFps.events[0].startSec, 5.0, 1e-9);
    EXPECT_LT(withoutFps.events[0].startFrame, 0);
    EXPECT_EQ(withoutFps.events[0].warnings.size(), 1U);

    // --fps supplies the missing rate.
    const GroundTruth withFps =
        touch_eval::ParseTouchCsvText(text, "inline", 60.0);
    ASSERT_EQ(withFps.events.size(), 1U);
    EXPECT_NEAR(withFps.events[0].startSec, 5.0, 1e-9);
    EXPECT_EQ(withFps.events[0].startFrame, 300);
    EXPECT_TRUE(withFps.events[0].warnings.empty());
}

TEST(TouchGroundTruthTest, ParsesTouchesCsvWithoutHeader) {
    // No header: the documented column order is assumed.
    const GroundTruth groundTruth = touch_eval::ParseTouchCsvText(
        "1,touch,3.25,3.25,1,left,,\n", "inline");
    ASSERT_EQ(groundTruth.events.size(), 1U);
    EXPECT_NEAR(groundTruth.events[0].startSec, 3.25, 1e-9);
    EXPECT_EQ(groundTruth.events[0].footHint, "left");
}

TEST(TouchGroundTruthTest, ConvertsLabelToCsvAndBack) {
    const GroundTruth label = touch_eval::ParseTouchLabelText(
        "00:07S（2次）\n00:34S - 00:35 S (1次)\n", "inline");
    const std::string path = "/tmp/nexusflow_touch_eval_csv_test.csv";

    std::string error;
    ASSERT_TRUE(touch_eval::ConvertTouchLabelToCsv(label, path, &error)) << error;
    GroundTruth csv;
    ASSERT_TRUE(touch_eval::ParseTouchCsvFile(path, &csv, &error)) << error;
    std::remove(path.c_str());

    // "(2次)" becomes two one-touch rows, both flagged for a manual split.
    ASSERT_EQ(csv.events.size(), 3U);
    EXPECT_NEAR(csv.events[0].startSec, 7.0, 1e-9);
    EXPECT_NEAR(csv.events[1].startSec, 7.0, 1e-9);
    EXPECT_EQ(csv.events[0].count, 1);
    EXPECT_NE(csv.events[0].note.find("SPLIT ME"), std::string::npos);
    EXPECT_NEAR(csv.events[2].startSec, 34.0, 1e-9);
    EXPECT_NEAR(csv.events[2].endSec, 35.0, 1e-9);
    EXPECT_EQ(csv.TouchCount(), 3);
}

TEST(TouchGroundTruthTest, ParsesLowQualityColumnAndKeepsTheFileOrder) {
    const std::string text =
        "# point_window_sec: 1.00\n"
        "index,type,t_start_sec,t_end_sec,count,foot,player,low_quality_data,note\n"
        "7,touch,5.00,5.00,1,,,yes,blurred\n"
        "9,no_touch,8.00,9.00,0,,,,negative\n"
        "11,touch,12.00,12.00,1,,,no,readable\n";

    const GroundTruth groundTruth = touch_eval::ParseTouchCsvText(text, "inline");
    ASSERT_TRUE(groundTruth.warnings.empty());
    ASSERT_EQ(groundTruth.events.size(), 2U);
    ASSERT_EQ(groundTruth.noTouch.size(), 1U);

    EXPECT_TRUE(groundTruth.events[0].lowQuality);
    // `no` is a usable "not flagged", not an unreadable value.
    EXPECT_FALSE(groundTruth.events[1].lowQuality);
    EXPECT_TRUE(groundTruth.events[1].warnings.empty());
    EXPECT_FALSE(groundTruth.noTouch[0].lowQuality);
    EXPECT_EQ(groundTruth.LowQualityRows(), 1);
    EXPECT_EQ(groundTruth.LowQualityTouchCount(), 1);

    // The index column is what the annotator reads, so the rows keep it and the
    // negative span in the middle does not push the later touches down.
    EXPECT_EQ(groundTruth.events[0].index, 7);
    EXPECT_EQ(groundTruth.noTouch[0].index, 9);
    EXPECT_EQ(groundTruth.events[1].index, 11);
    EXPECT_EQ(groundTruth.events[0].sourceRow, 1);
    EXPECT_EQ(groundTruth.noTouch[0].sourceRow, 2);
    EXPECT_EQ(groundTruth.events[1].sourceRow, 3);

    // Writing it back keeps the annotator's row order and renumbers the rows.
    const std::string path = "/tmp/nexusflow_touch_eval_lowq.csv";
    std::string error;
    ASSERT_TRUE(touch_eval::WriteTouchCsv(groundTruth, path, &error)) << error;
    GroundTruth reloaded;
    ASSERT_TRUE(touch_eval::ParseTouchCsvFile(path, &reloaded, &error)) << error;
    std::remove(path.c_str());

    ASSERT_EQ(reloaded.events.size(), 2U);
    ASSERT_EQ(reloaded.noTouch.size(), 1U);
    EXPECT_EQ(reloaded.events[0].index, 1);
    EXPECT_EQ(reloaded.noTouch[0].index, 2);
    EXPECT_EQ(reloaded.events[1].index, 3);
    EXPECT_TRUE(reloaded.events[0].lowQuality);
    EXPECT_FALSE(reloaded.events[1].lowQuality);
    EXPECT_EQ(reloaded.LowQualityRows(), 1);
}

TEST(TouchGroundTruthTest, RenumberTidiesAHandEditedFileAndKeepsABackup) {
    const std::string path = "/tmp/nexusflow_touch_eval_renumber.csv";
    const std::string original =
        "# touch annotations v2\n"
        "index,type,t_start_sec,t_end_sec,count,foot,player,note\n"
        "4,touch,5.00,5.00,1,,,first\n"
        "4,touch,8.00,9.00,1,,,duplicated index\n"
        ",no_touch,20.00,21.00,0,,,blank index\n";
    {
        std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
        ASSERT_TRUE(file.is_open());
        file << original;
    }

    int rows = 0;
    std::string error;
    ASSERT_TRUE(touch_eval::RenumberTouchCsvFile(path, &rows, &error)) << error;
    EXPECT_EQ(rows, 3);

    GroundTruth reloaded;
    ASSERT_TRUE(touch_eval::ParseTouchCsvFile(path, &reloaded, &error)) << error;
    ASSERT_EQ(reloaded.events.size(), 2U);
    ASSERT_EQ(reloaded.noTouch.size(), 1U);
    // Every row gets its own number, in the order the annotator wrote them.
    EXPECT_EQ(reloaded.events[0].index, 1);
    EXPECT_EQ(reloaded.events[1].index, 2);
    EXPECT_EQ(reloaded.noTouch[0].index, 3);
    EXPECT_NE(reloaded.events[1].note.find("duplicated index"), std::string::npos);

    // The previous content survives next to the file.
    std::ifstream backup((path + ".bak").c_str());
    ASSERT_TRUE(backup.is_open());
    std::ostringstream buffer;
    buffer << backup.rdbuf();
    EXPECT_EQ(buffer.str(), original);

    std::remove(path.c_str());
    std::remove((path + ".bak").c_str());
}

TEST(TouchGroundTruthTest, RenumberRefusesAFileWithUnreadableRows) {
    const std::string path = "/tmp/nexusflow_touch_eval_renumber_bad.csv";
    const std::string original =
        "index,type,t_start_sec,t_end_sec,count,foot,player,note\n"
        "1,touch,5.00,5.00,1,,,fine\n"
        "2,touch,oops,6.00,1,,,broken start time\n";
    {
        std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
        ASSERT_TRUE(file.is_open());
        file << original;
    }

    int rows = 0;
    std::string error;
    // Rewriting would drop the unreadable row: refuse and leave the file alone.
    EXPECT_FALSE(touch_eval::RenumberTouchCsvFile(path, &rows, &error));
    EXPECT_NE(error.find("refusing to rewrite"), std::string::npos);
    std::ifstream check(path.c_str());
    std::ostringstream buffer;
    buffer << check.rdbuf();
    EXPECT_EQ(buffer.str(), original);

    std::remove(path.c_str());
    std::remove((path + ".bak").c_str());
}

TEST(TouchGroundTruthTest, CsvWriteKeepsNegativeSpans) {
    GroundTruth groundTruth;
    groundTruth.sourceKind = "csv";
    GroundTruthEvent touch;
    touch.startSec = 5.0;
    touch.endSec = 5.0;
    touch.count = 1;
    groundTruth.events.push_back(touch);
    GroundTruthEvent negative;
    negative.startSec = 20.0;
    negative.endSec = 24.5;
    negative.count = 0;
    negative.note = "球出界";
    groundTruth.noTouch.push_back(negative);

    const std::string path = "/tmp/nexusflow_touch_eval_csv_negative.csv";
    std::string error;
    ASSERT_TRUE(touch_eval::WriteTouchCsv(groundTruth, path, &error)) << error;
    GroundTruth reloaded;
    ASSERT_TRUE(touch_eval::ParseTouchCsvFile(path, &reloaded, &error)) << error;
    std::remove(path.c_str());

    ASSERT_EQ(reloaded.events.size(), 1U);
    ASSERT_EQ(reloaded.noTouch.size(), 1U);
    EXPECT_NEAR(reloaded.noTouch[0].startSec, 20.0, 1e-9);
    EXPECT_NEAR(reloaded.noTouch[0].endSec, 24.5, 1e-9);
    EXPECT_EQ(reloaded.noTouch[0].note, "球出界");
    EXPECT_EQ(reloaded.TouchCount(), 1);
}

TEST(TouchDetectorTest, ResetClearsTheOpenCluster) {
    const TouchDetectorParams params;
    TouchDetector detector(params);
    detector.Push(DistanceFrame(0, 20.0f));
    detector.Reset();
    EXPECT_TRUE(detector.Flush().empty());
}

TEST(TouchDetectorTest, RefractoryWindowMergesTheSameTouchReportedTwice) {
    TouchDetectorParams base;
    base.refractoryFrames = 0;

    // Without the window, six frames apart is two touches.
    {
        TouchDetector detector(base);
        std::vector<FrameEvidence> frames = TwoContactsScenario(6, base);
        int events = 0;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            events += static_cast<int>(detector.PushEvidence(frames[i]).size());
        }
        events += static_cast<int>(detector.Flush().size());
        EXPECT_EQ(events, 2);
        EXPECT_EQ(detector.SuppressedByRefractory(), 0);
    }
    // A 12 frame window (0.2 s at 60 fps) merges them into one.
    {
        TouchDetectorParams params = base;
        params.refractoryFrames = 12;
        TouchDetector detector(params);
        std::vector<FrameEvidence> frames = TwoContactsScenario(6, params);
        int events = 0;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            events += static_cast<int>(detector.PushEvidence(frames[i]).size());
        }
        events += static_cast<int>(detector.Flush().size());
        EXPECT_EQ(events, 1);
        EXPECT_EQ(detector.SuppressedByRefractory(), 1);
    }
}

TEST(TouchDetectorTest, SpeedGateKeepsAKickedBallAndDropsARollingOne) {
    const TouchDetectorParams base;

    const auto countEvents = [](const TouchDetectorParams& params,
                                float stepPx) {
        TouchDetector detector(params);
        const std::vector<FrameEvidence> frames = ApproachScenario(stepPx, params);
        int events = 0;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            events += static_cast<int>(detector.PushEvidence(frames[i]).size());
        }
        events += static_cast<int>(detector.Flush().size());
        return events;
    };

    // No gate: both the kicked ball (20 px/frame) and the slowly rolling one
    // (2 px/frame) look like a touch.
    EXPECT_EQ(countEvents(base, 20.0f), 1);
    EXPECT_EQ(countEvents(base, 2.0f), 1);

    // With the gate, only the sharp speed change counts. The kicked ball goes
    // from 1200 px/s to nothing, the rolling one from 120 px/s to nothing.
    TouchDetectorParams gated = base;
    gated.minSpeedChangePxPerSec = 500.0f;
    EXPECT_EQ(countEvents(gated, 20.0f), 1);
    EXPECT_EQ(countEvents(gated, 2.0f), 0);

    TouchDetector counter(gated);
    const std::vector<FrameEvidence> rolling = ApproachScenario(2.0f, gated);
    for (std::size_t i = 0; i < rolling.size(); ++i) {
        counter.PushEvidence(rolling[i]);
    }
    counter.Flush();
    EXPECT_EQ(counter.Aggregator().RejectedBySpeedGate(), 1);
}

TEST(TouchDetectorTest, TrackedBallOnlyGateDropsCandidateTouches) {
    TouchDetectorParams params;
    params.requireTrackedBall = true;
    TouchDetector detector(params);

    // Same geometry, but the touch comes from an untracked candidate ball.
    TouchFrameInput frame = MakeFrame(0, 0.0);
    frame.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));
    frame.balls.push_back(MakeBall(110.0f, 100.0f, 0.9f, -1));  // track_id -1
    TouchFrameInput second = MakeFrame(1, 1.0 / 60.0);
    second.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));
    second.balls.push_back(MakeBall(110.0f, 100.0f, 0.9f, -1));

    detector.Push(frame);
    detector.Push(second);
    const std::vector<TouchEvent> events = detector.Flush();
    EXPECT_TRUE(events.empty());
    EXPECT_EQ(detector.Aggregator().RejectedByBallGate(), 1);

    // The same frames with a followed ball do produce a touch.
    TouchFrameInput tracked = MakeFrame(0, 0.0);
    tracked.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));
    tracked.balls.push_back(MakeBall(110.0f, 100.0f, 0.9f, 3));
    TouchFrameInput trackedSecond = MakeFrame(1, 1.0 / 60.0);
    trackedSecond.persons.push_back(MakePerson(1, 600.0f, 100.0f, 100.0f));
    trackedSecond.balls.push_back(MakeBall(110.0f, 100.0f, 0.9f, 3));
    TouchDetector trackedDetector(params);
    trackedDetector.Push(tracked);
    trackedDetector.Push(trackedSecond);
    EXPECT_EQ(trackedDetector.Flush().size(), 1U);
}

TEST(TouchMetricsTest, MatchesEventsCountsAndCalibratesThreshold) {
    // At the default ratio the enter threshold is 32.4 px (frame term) and the
    // 600 px player allows 36 px, so a 25 px approach is a contact.
    const touch_eval::EvaluationOptions options =
        touch_eval::DefaultEvaluationOptions();
    const TouchDetectorParams params = options.params;
    const float contactDistancePx = 25.0f;

    std::vector<FrameEvidence> frames;
    AppendTouch(&frames, 10.0, contactDistancePx, params);
    AppendTouch(&frames, 20.0, contactDistancePx, params);
    AppendTouch(&frames, 20.9, contactDistancePx, params);
    AppendTouch(&frames, 40.0, contactDistancePx, params);

    // 10.0 s: one touch; 20-21 s: two touches; 40.0 s: one touch.
    const GroundTruth groundTruth = touch_eval::ParseTouchLabelText(
        "00:10S (1次)\n00:20S - 00:21S (2次)\n00:40S (1次)\n", "inline");
    ASSERT_EQ(groundTruth.TouchCount(), 4);

    const touch_eval::EvaluationResult result =
        touch_eval::EvaluateFrames(frames, groundTruth, options);
    ASSERT_EQ(result.events.size(), 4U);

    // Primary (count aware): the "2次" annotation at 20-21 s hands out two
    // slots, so both events inside it are found and nothing is left over.
    EXPECT_NEAR(result.instance.pointWindowSec, 1.0, 1e-9);
    EXPECT_EQ(result.instance.capacity, 4);
    EXPECT_EQ(result.instance.truePositives, 4);
    EXPECT_EQ(result.instance.falsePositives, 0);
    EXPECT_EQ(result.instance.falseNegatives, 0);
    EXPECT_NEAR(result.instance.f1, 1.0, 1e-9);
    EXPECT_EQ(result.instance.windowsWithEvent, 3);
    EXPECT_EQ(result.instance.windowsExact, 3);
    EXPECT_EQ(result.eventGtIndex[0], 0);
    EXPECT_EQ(result.eventGtIndex[1], 1);
    EXPECT_EQ(result.eventGtIndex[2], 1);
    EXPECT_EQ(result.eventGtIndex[3], 2);
    EXPECT_EQ(result.falsePositives.unannotated, 0);

    // Auxiliary (one-to-one): only one of the two events can claim the 20-21 s
    // annotation, so the second one is a false positive. This is the metric
    // that earlier runs reported.
    EXPECT_EQ(result.primary.truePositives, 3);
    EXPECT_EQ(result.primary.falsePositives, 1);
    EXPECT_EQ(result.primary.falseNegatives, 0);
    EXPECT_NEAR(result.primary.recall, 1.0, 1e-9);
    EXPECT_NEAR(result.primary.precision, 0.75, 1e-9);

    // The count view keeps both touches of the crowded interval.
    EXPECT_EQ(result.counts.exactRecords, 3);
    EXPECT_EQ(result.counts.predictedInWindows, 4);
    EXPECT_EQ(result.counts.gtTouches, 4);
    EXPECT_NEAR(result.counts.meanAbsCountError, 0.0, 1e-9);

    // Every window had a ball and usable feet.
    EXPECT_EQ(result.diagnostics.records, 3);
    EXPECT_EQ(result.diagnostics.withUsableBall, 3);
    EXPECT_EQ(result.diagnostics.withUsableFeet, 3);

    // Under the primary metric every window is served and nothing is left
    // over, so there is neither a miss to explain nor a false positive.
    EXPECT_TRUE(result.failures.empty());

    // Calibration: 25 px is missed while the enter threshold stays at 18 px
    // (ratio 0.005) or 21.6 px (ratio 0.03), and caught from 0.045 up.
    EXPECT_EQ(result.sweep.size(), options.sweepRatios.size());
    EXPECT_EQ(result.sweep[0].truePositives, 0);
    EXPECT_EQ(result.sweep[0].falseNegatives, 3);
    EXPECT_NEAR(result.sweep[0].instanceF1, 0.0, 1e-9);
    EXPECT_EQ(result.sweep[5].truePositives, 3);
    EXPECT_EQ(result.sweep.back().truePositives, 3);
    EXPECT_NEAR(result.sweep[5].thresholdPxRef, 32.4, 1e-3);
    EXPECT_NEAR(result.sweep[5].instanceF1, 1.0, 1e-9);
}

TEST(TouchMetricsTest, ScoresTwiceWhenRowsAreFlaggedLowQuality) {
    touch_eval::EvaluationOptions options =
        touch_eval::DefaultEvaluationOptions();
    // The refractory window is a different feature; here it would merge the two
    // detections of the flagged window before the views could differ.
    options.params.refractoryFrames = 0;
    const TouchDetectorParams params = options.params;

    const GroundTruth groundTruth = touch_eval::ParseTouchCsvText(
        "# point_window_sec: 1.00\n"
        "index,type,t_start_sec,t_end_sec,count,foot,player,low_quality_data,note\n"
        "1,touch,10.00,10.00,1,,,,readable\n"
        "2,touch,20.00,20.00,1,,,yes,太模糊了人员\n"
        "3,touch,40.00,40.00,1,,,,readable\n",
        "inline");
    ASSERT_EQ(groundTruth.TouchCount(), 3);
    ASSERT_EQ(groundTruth.LowQualityRows(), 1);

    std::vector<FrameEvidence> frames;
    AppendTouch(&frames, 10.0, 25.0f, params);
    AppendTouch(&frames, 20.0, 25.0f, params);  // inside the flagged window
    AppendTouch(&frames, 20.4, 25.0f, params);  // second one, same window
    AppendTouch(&frames, 40.0, 25.0f, params);

    const touch_eval::EvaluationResult result =
        touch_eval::EvaluateFrames(frames, groundTruth, options);
    ASSERT_EQ(result.events.size(), 4U);

    // Full view: the flagged row still claims one slot, so the second detection
    // in it is a false positive.
    EXPECT_EQ(result.instance.capacity, 3);
    EXPECT_EQ(result.instance.truePositives, 3);
    EXPECT_EQ(result.instance.falsePositives, 1);
    EXPECT_EQ(result.instance.falseNegatives, 0);
    EXPECT_NEAR(result.instance.f1, 0.857, 1e-3);
    EXPECT_EQ(result.counts.records, 3);

    // Second view: the flagged row goes, and so do the two detections that only
    // it could explain - neither a hit nor a false positive.
    const touch_eval::LowQualityScore& clean = result.lowQuality;
    EXPECT_EQ(clean.rows, 1);
    EXPECT_EQ(clean.touches, 1);
    EXPECT_EQ(clean.excludedEvents, 2);
    EXPECT_EQ(clean.instance.capacity, 2);
    EXPECT_EQ(clean.instance.truePositives, 2);
    EXPECT_EQ(clean.instance.falsePositives, 0);
    EXPECT_EQ(clean.instance.falseNegatives, 0);
    EXPECT_NEAR(clean.instance.f1, 1.0, 1e-9);
    EXPECT_EQ(clean.counts.records, 2);
    EXPECT_EQ(clean.tolerances.size(), options.tolerances.size());
    EXPECT_NEAR(clean.primary.f1, 1.0, 1e-9);
    // The flagged window was perfectly readable, which the report says out loud.
    EXPECT_EQ(clean.windowsWithUsableBall, 1);
    EXPECT_EQ(clean.windowsWithUsableFeet, 1);
    EXPECT_EQ(clean.personHeightPx.samples, 1);
}

TEST(TouchMetricsTest, ClassifiesFalsePositivesBySource) {
    touch_eval::EvaluationOptions options =
        touch_eval::DefaultEvaluationOptions();
    // The classification looks at where an event sits, so the refractory window
    // (which would merge two nearby events) has to be out of the way here.
    options.params.refractoryFrames = 0;
    const TouchDetectorParams params = options.params;

    // A v2 CSV source: a point annotation stands for half a second, and the
    // 20-25 s span is a confirmed negative sample.
    const GroundTruth groundTruth = touch_eval::ParseTouchCsvText(
        "index,type,t_start_sec,t_end_sec,count,foot,player,note\n"
        "1,touch,10.00,10.00,1,,,\n"
        "2,no_touch,20.00,25.00,0,,,\n",
        "inline");

    std::vector<FrameEvidence> frames;
    AppendTouch(&frames, 10.0, 25.0f, params);   // in the window: the one true hit
    AppendTouch(&frames, 10.2, 25.0f, params);   // same touch, another frame
    AppendTouch(&frames, 10.1, 25.0f, params, 600.0f, 2);  // same window, but the
                                                           // window holds one slot
    AppendTouch(&frames, 11.0, 25.0f, params);   // just past the window
    AppendTouch(&frames, 22.0, 25.0f, params);   // inside the no_touch span
    AppendTouch(&frames, 200.0, 25.0f, params);  // nothing annotated nearby

    const touch_eval::EvaluationResult result =
        touch_eval::EvaluateFrames(frames, groundTruth, options);
    ASSERT_EQ(result.events.size(), 6U);
    EXPECT_NEAR(result.instance.pointWindowSec, 0.5, 1e-9);
    EXPECT_EQ(result.instance.truePositives, 1);
    EXPECT_EQ(result.instance.falsePositives, 5);
    EXPECT_EQ(result.instance.capacity, 1);

    EXPECT_EQ(result.falsePositives.duplicates, 1);
    EXPECT_EQ(result.falsePositives.inWindowNoSlot, 1);
    EXPECT_EQ(result.falsePositives.insideNoTouch, 1);
    EXPECT_EQ(result.falsePositives.adjacent, 1);
    EXPECT_EQ(result.falsePositives.unannotated, 1);
    EXPECT_EQ(result.falsePositives.examples.size(), 5U);
}

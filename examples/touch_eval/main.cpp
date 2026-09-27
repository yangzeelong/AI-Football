#include "ObservationJsonlReader.hpp"

#include <touch_eval/TouchDetector.hpp>
#include <touch_eval/TouchGroundTruth.hpp>
#include <touch_eval/TouchMetrics.hpp>

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string observationsPath;
    std::string labelPath;
    std::string touchesCsvPath;
    std::string gtJsonPath;
    std::string outputDir;
    std::string dumpGtPath;
    std::string dumpGtCsvPath;
    /// Frame rate used to read frame based annotation rows (--fps); 0 = use the
    /// `# fps:` hint in the file.
    double fps = 0.0;
    int frameStride = 1;
    bool writeCsv = true;
    bool runSweep = true;
    bool renumberTouches = false;
    bool quiet = false;
    bool showHelp = false;
    touch_eval::EvaluationOptions evaluation;
};

void PrintUsage() {
    std::cout <<
        "touch_eval - score detected ball touches against the annotations\n"
        "\n"
        "usage: touch_eval [--observations <observations.jsonl>]\n"
        "                  (--label <file> | --touches_csv <file.csv> | --gt_json <file.json>)\n"
        "                  [--output_dir <dir>]\n"
        "                  [--dump_gt <file.json>] [--dump_gt_csv <file.csv>]\n"
        "                  [--fps <rate>]\n"
        "                  [--tolerance a,b,c] [--primary_tolerance s]\n"
        "                  [--contact_ratio r] [--person_height_ratio r]\n"
        "                  [--min_contact_px px] [--max_contact_px px]\n"
        "                  [--release_ratio r] [--kp_conf c] [--ball_conf c]\n"
        "                  [--min_contact_frames n] [--max_gap_frames n]\n"
        "                  [--min_split_frames n] [--min_rise_frames n]\n"
        "                  [--refractory_frames n] [--min_speed_change px_s]\n"
        "                  [--min_speed_ratio r] [--tracked_ball_only 0|1]\n"
        "                  [--window s] [--frame_stride n]\n"
        "                  [--point_window s] [--window_pad s]\n"
        "                  [--adjacency s] [--duplicate_sec s]\n"
        "                  [--renumber_touches]\n"
        "                  [--no_csv] [--no_sweep] [--quiet]\n"
        "\n"
        "--label picks its parser from the extension (.label / .csv / .json), so a\n"
        "v2 CSV can be passed as --label directly. Without --observations the tool\n"
        "only converts the annotations (--dump_gt / --dump_gt_csv) and stops.\n"
        "--dump_gt_csv on a legacy .label expands \"(N次)\" into one row per touch\n"
        "and marks those rows SPLIT ME.\n"
        "--renumber_touches rewrites a v2 CSV in place with sequential indices and\n"
        "the canonical columns, keeping the row order and a <file>.bak copy.\n"
        "\n"
        "Rows flagged in `low_quality_data` are scored twice: the report carries\n"
        "the full result and the result with those rows and the detections they\n"
        "alone explain taken out.\n"
        "\n"
        "Outputs (in --output_dir): touch_report.txt, touch_report.json,\n"
        "touch_events.jsonl and contact_signal.csv.\n";
}

bool NextValue(int argc, char** argv, int* index, std::string* value) {
    if (*index + 1 >= argc) return false;
    ++(*index);
    *value = argv[*index];
    return true;
}

bool ParseToleranceList(const std::string& text, std::vector<double>* values) {
    values->clear();
    std::istringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) continue;
        values->push_back(std::atof(item.c_str()));
    }
    return !values->empty();
}

/// Directory part of a path, "." when there is none.
std::string DirNameOf(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

std::string JoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir == ".") return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

/// Creates `path` and its parents. Returns false only when the directory still
/// does not exist afterwards.
bool EnsureDirectory(const std::string& path) {
    if (path.empty() || path == "." || path == "/") return true;
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode) != 0;
    }
    const std::size_t slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        if (!EnsureDirectory(path.substr(0, slash))) return false;
    }
    return mkdir(path.c_str(), 0777) == 0;
}

bool ParseOptions(int argc, char** argv, Options* options, std::string* error) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        std::string value;
        if (argument == "--help" || argument == "-h") {
            options->showHelp = true;
            continue;
        }
        if (argument == "--renumber_touches") {
            options->renumberTouches = true;
            continue;
        }
        if (argument == "--no_csv") { options->writeCsv = false; continue; }
        if (argument == "--no_sweep") { options->runSweep = false; continue; }
        if (argument == "--quiet") { options->quiet = true; continue; }

        const bool wantsValue = (argument == "--observations") ||
            (argument == "--label") || (argument == "--touches_csv") ||
            (argument == "--gt_json") ||
            (argument == "--output_dir") || (argument == "--dump_gt") ||
            (argument == "--dump_gt_csv") ||
            (argument == "--tolerance") || (argument == "--primary_tolerance") ||
            (argument == "--contact_ratio") || (argument == "--person_height_ratio") ||
            (argument == "--min_contact_px") || (argument == "--max_contact_px") ||
            (argument == "--release_ratio") || (argument == "--kp_conf") ||
            (argument == "--ball_conf") || (argument == "--min_contact_frames") ||
            (argument == "--max_gap_frames") || (argument == "--min_split_frames") ||
            (argument == "--min_rise_frames") || (argument == "--window") ||
            (argument == "--point_window") || (argument == "--window_pad") ||
            (argument == "--adjacency") || (argument == "--duplicate_sec") ||
            (argument == "--fps") ||
            (argument == "--refractory_frames") ||
            (argument == "--min_speed_change") ||
            (argument == "--min_speed_ratio") ||
            (argument == "--tracked_ball_only") ||
            (argument == "--frame_stride");
        if (!wantsValue) {
            *error = "unknown argument '" + argument + "'";
            return false;
        }
        if (!NextValue(argc, argv, &i, &value)) {
            *error = "missing value for '" + argument + "'";
            return false;
        }

        touch_eval::TouchDetectorParams& params = options->evaluation.params;
        if (argument == "--observations") options->observationsPath = value;
        else if (argument == "--label") options->labelPath = value;
        else if (argument == "--touches_csv") options->touchesCsvPath = value;
        else if (argument == "--gt_json") options->gtJsonPath = value;
        else if (argument == "--output_dir") options->outputDir = value;
        else if (argument == "--dump_gt") options->dumpGtPath = value;
        else if (argument == "--dump_gt_csv") options->dumpGtCsvPath = value;
        else if (argument == "--fps") options->fps = std::atof(value.c_str());
        else if (argument == "--tolerance") {
            if (!ParseToleranceList(value, &options->evaluation.tolerances)) {
                *error = "cannot parse --tolerance '" + value + "'";
                return false;
            }
        }
        else if (argument == "--primary_tolerance") options->evaluation.primaryToleranceSec = std::atof(value.c_str());
        else if (argument == "--contact_ratio") params.contactDistanceRatio = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--person_height_ratio") params.personHeightRatio = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--min_contact_px") params.minContactPx = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--max_contact_px") params.maxContactPx = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--release_ratio") params.releaseRatio = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--kp_conf") params.keypointConfidence = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--ball_conf") params.ballMinConfidence = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--min_contact_frames") params.minContactFrames = std::atoi(value.c_str());
        else if (argument == "--max_gap_frames") params.maxGapFrames = std::atoi(value.c_str());
        else if (argument == "--min_split_frames") params.minSplitFrames = std::atoi(value.c_str());
        else if (argument == "--min_rise_frames") params.minRiseFrames = std::atoi(value.c_str());
        else if (argument == "--refractory_frames") params.refractoryFrames = std::atoi(value.c_str());
        else if (argument == "--min_speed_change") params.minSpeedChangePxPerSec = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--min_speed_ratio") params.minSpeedRatio = static_cast<float>(std::atof(value.c_str()));
        else if (argument == "--tracked_ball_only") params.requireTrackedBall = std::atoi(value.c_str()) != 0;
        else if (argument == "--window") options->evaluation.windowSec = std::atof(value.c_str());
        else if (argument == "--point_window") options->evaluation.pointWindowSec = std::atof(value.c_str());
        else if (argument == "--window_pad") options->evaluation.windowPadSec = std::atof(value.c_str());
        else if (argument == "--adjacency") options->evaluation.adjacencySec = std::atof(value.c_str());
        else if (argument == "--duplicate_sec") options->evaluation.duplicateSec = std::atof(value.c_str());
        else if (argument == "--frame_stride") options->frameStride = std::atoi(value.c_str());
    }

    if (options->showHelp) return true;

    const bool hasLabel = !options->labelPath.empty();
    const bool hasCsv = !options->touchesCsvPath.empty();
    const bool hasJson = !options->gtJsonPath.empty();
    const int sources = (hasLabel ? 1 : 0) + (hasCsv ? 1 : 0) + (hasJson ? 1 : 0);
    if (sources == 0) {
        *error = "one of --label, --touches_csv or --gt_json is required";
        return false;
    }
    if (sources > 1) {
        *error = "--label, --touches_csv and --gt_json are mutually exclusive";
        return false;
    }
    // Without observations the tool is a converter: it only rewrites the
    // annotations, so requiring the frame file would be pointless.
    const bool conversionOnly = options->observationsPath.empty();
    if (conversionOnly && options->dumpGtPath.empty() &&
        options->dumpGtCsvPath.empty() && !options->renumberTouches) {
        *error = "--observations is required unless --dump_gt, --dump_gt_csv or "
                 "--renumber_touches is given";
        return false;
    }
    if (options->frameStride < 1) options->frameStride = 1;
    if (options->outputDir.empty()) {
        options->outputDir = options->observationsPath.empty()
            ? DirNameOf(hasCsv ? options->touchesCsvPath
                               : hasJson ? options->gtJsonPath : options->labelPath)
            : DirNameOf(options->observationsPath);
    }
    return true;
}

const char* LevelName(touch_eval::ContactLevel level) {
    switch (level) {
        case touch_eval::ContactLevel::Contact: return "contact";
        case touch_eval::ContactLevel::Band: return "band";
        case touch_eval::ContactLevel::None: break;
    }
    return "none";
}

bool WriteContactSignalCsv(const std::vector<touch_eval::FrameContact>& contacts,
                           const std::string& path, std::string* error) {
    std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        *error = "cannot write '" + path + "'";
        return false;
    }
    file << "frame_index,timestamp_sec,level,person_track_id,foot,distance_px,"
            "threshold_px,ball_confidence,ball_tracked,ball_track_id,"
            "observed_balls,valid_foot_points,persons_with_feet,persons\n";
    for (std::size_t i = 0; i < contacts.size(); ++i) {
        const touch_eval::FrameContact& contact = contacts[i];
        file << contact.frameIndex << ',' << contact.timestampSec << ','
             << LevelName(contact.level) << ',';
        if (contact.hasBall) {
            file << contact.candidate.personTrackId << ','
                 << (contact.candidate.leftFoot ? "left" : "right") << ','
                 << contact.candidate.distancePx << ',' << contact.thresholdPx << ','
                 << contact.candidate.ballConfidence << ','
                 << (contact.candidate.ballTracked ? 1 : 0) << ','
                 << contact.candidate.ballTrackId << ',';
        } else {
            file << ",,,,,,";
        }
        file << contact.observedBallCount << ',' << contact.validFootPointCount << ','
             << contact.personsWithFeet << ',' << contact.personCount << '\n';
    }
    return true;
}

bool WriteTouchEventsJsonl(const touch_eval::EvaluationResult& result,
                           const std::string& path, std::string* error) {
    std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        *error = "cannot write '" + path + "'";
        return false;
    }
    for (std::size_t i = 0; i < result.events.size(); ++i) {
        const touch_eval::TouchEvent& event = result.events[i];
        const int gtIndex =
            i < result.eventGtIndex.size() ? result.eventGtIndex[i] : -1;
        const touch_eval::EventVerdict verdict =
            i < result.falsePositives.eventVerdict.size()
                ? result.falsePositives.eventVerdict[i]
                : touch_eval::EventVerdict::Assigned;
        file << "{\"type\":\"touch\",\"frame_index\":" << event.frameIndex
             << ",\"first_frame_index\":" << event.firstFrameIndex
             << ",\"timestamp_sec\":" << event.timestampSec
             << ",\"person_track_id\":" << event.personTrackId
             << ",\"foot\":\"" << (event.leftFoot ? "left" : "right") << "\""
             << ",\"distance_px\":" << event.distancePx
             << ",\"threshold_px\":" << event.thresholdPx
             << ",\"ball_confidence\":" << event.ballConfidence
             << ",\"ball_tracked\":" << (event.ballTracked ? "true" : "false")
             << ",\"ball_track_id\":" << event.ballTrackId
             << ",\"contact_frames\":" << event.contactFrames
             << ",\"speed_before_px_per_sec\":" << event.speedBeforePxPerSec
             << ",\"speed_after_px_per_sec\":" << event.speedAfterPxPerSec
             << ",\"gt_index\":" << gtIndex
             << ",\"matched\":" << (gtIndex >= 0 ? "true" : "false")
             << ",\"verdict\":\""
             << touch_eval::EventVerdictName(verdict) << "\"}\n";
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    std::string error;
    if (!ParseOptions(argc, argv, &options, &error)) {
        std::cerr << "touch_eval: " << error << "\n";
        PrintUsage();
        return 1;
    }
    if (options.showHelp) {
        PrintUsage();
        return 0;
    }

    touch_eval::GroundTruth groundTruth;
    bool groundTruthFromJson = false;
    if (!options.labelPath.empty()) {
        // The extension picks the parser, so a v2 CSV can be passed as --label.
        if (!touch_eval::ParseTouchAnnotationFile(options.labelPath, &groundTruth,
                                                  &error, options.fps)) {
            std::cerr << "touch_eval: " << error << "\n";
            return 2;
        }
    } else if (!options.touchesCsvPath.empty()) {
        if (!touch_eval::ParseTouchCsvFile(options.touchesCsvPath, &groundTruth,
                                           &error, options.fps)) {
            std::cerr << "touch_eval: " << error << "\n";
            return 2;
        }
    } else {
        if (!touch_eval::ReadGroundTruthJson(options.gtJsonPath, &groundTruth, &error)) {
            std::cerr << "touch_eval: " << error << "\n";
            return 2;
        }
        groundTruthFromJson = true;
        if (groundTruth.source.empty()) groundTruth.source = options.gtJsonPath;
        touch_eval::ApplyFpsToGroundTruth(&groundTruth, options.fps);
    }
    if (groundTruth.Empty()) {
        std::cerr << "touch_eval: ground truth has no usable event\n";
        return 2;
    }
    if (!options.quiet) {
        std::cout << "annotations: " << groundTruth.sourceKind << " '"
                  << groundTruth.source << "' - " << groundTruth.events.size()
                  << " rows, " << groundTruth.TouchCount() << " touches, "
                  << groundTruth.noTouch.size() << " no_touch span(s), window "
                  << groundTruth.defaultPointWindowSec << " s for a point";
        if (groundTruth.fps > 0.0) {
            std::cout << ", fps " << groundTruth.fps << " (frames known)";
        }
        std::cout << "\n";
    }

    if (options.renumberTouches) {
        const std::string path = !options.touchesCsvPath.empty()
            ? options.touchesCsvPath
            : (!options.labelPath.empty() ? options.labelPath
                                          : options.gtJsonPath);
        if (groundTruth.sourceKind != "csv") {
            std::cerr << "touch_eval: --renumber_touches needs a v2 CSV, but '"
                      << path << "' is a " << groundTruth.sourceKind
                      << " source\n";
            return 2;
        }
        int rows = 0;
        if (!touch_eval::RenumberTouchCsvFile(path, &rows, &error)) {
            std::cerr << "touch_eval: " << error << "\n";
            return 2;
        }
        if (!options.quiet) {
            std::cout << "renumbered     : " << rows << " row(s) in " << path
                      << " (previous content kept as " << path << ".bak)\n";
        }
        // Re-read the tidied file, so everything below - the dumped ground
        // truth included - describes what is on disk now.
        if (!touch_eval::ParseTouchCsvFile(path, &groundTruth, &error,
                                           options.fps)) {
            std::cerr << "touch_eval: " << error << "\n";
            return 2;
        }
    }

    if (!options.dumpGtPath.empty()) {
        if (!touch_eval::WriteGroundTruthJson(groundTruth, options.dumpGtPath, &error)) {
            std::cerr << "touch_eval: " << error << "\n";
            return 2;
        }
        if (!options.quiet) {
            std::cout << "wrote ground truth: " << options.dumpGtPath << " ("
                      << groundTruth.events.size() << " annotations, "
                      << groundTruth.TouchCount() << " touches, "
                      << groundTruth.warnings.size() << " file warnings)\n";
        }
    }

    if (!options.dumpGtCsvPath.empty()) {
        // A legacy .label is expanded to one row per touch; a CSV source is
        // written back faithfully.
        const bool converted = groundTruth.sourceKind != "csv"
            ? touch_eval::ConvertTouchLabelToCsv(groundTruth,
                                                 options.dumpGtCsvPath, &error)
            : touch_eval::WriteTouchCsv(groundTruth, options.dumpGtCsvPath,
                                        &error);
        if (!converted) {
            std::cerr << "touch_eval: " << error << "\n";
            return 2;
        }
        if (!options.quiet) {
            std::cout << "wrote touch csv : " << options.dumpGtCsvPath << " ("
                      << groundTruth.TouchCount() << " touch rows"
                      << (groundTruth.sourceKind == "csv" ? "" : ", counts expanded")
                      << ")\n";
        }
    }

    if (options.observationsPath.empty()) {
        if (!options.quiet) std::cout << "no --observations: conversion only\n";
        return 0;
    }

    touch_eval::ObservationJsonlReader reader;
    if (!reader.Open(options.observationsPath, &error)) {
        std::cerr << "touch_eval: " << error << "\n";
        return 2;
    }

    std::vector<touch_eval::FrameEvidence> frames;
    touch_eval::TouchFrameInput frame;
    while (reader.Next(&frame)) {
        if (options.frameStride > 1 &&
            (frame.frameIndex % static_cast<uint64_t>(options.frameStride)) != 0) {
            continue;
        }
        frames.push_back(
            touch_eval::BuildFrameEvidence(frame, options.evaluation.params));
    }
    if (frames.empty()) {
        std::cerr << "touch_eval: observations contained no frame\n";
        return 2;
    }

    touch_eval::EvaluationResult result =
        touch_eval::EvaluateFrames(frames, groundTruth, options.evaluation);

    touch_eval::ReportContext context;
    context.metadata = reader.Metadata();
    context.observationsPath = options.observationsPath;
    context.groundTruthSource = groundTruth.source;
    context.groundTruth = groundTruth;
    context.groundTruthFromJson = groundTruthFromJson;
    context.options = options.evaluation;

    if (!EnsureDirectory(options.outputDir)) {
        std::cerr << "touch_eval: cannot create output directory '"
                  << options.outputDir << "'\n";
        return 2;
    }

    const std::string reportTextPath =
        JoinPath(options.outputDir, "touch_report.txt");
    const std::string reportJsonPath =
        JoinPath(options.outputDir, "touch_report.json");
    const std::string eventsPath =
        JoinPath(options.outputDir, "touch_events.jsonl");
    const std::string csvPath = JoinPath(options.outputDir, "contact_signal.csv");

    const std::string reportText =
        touch_eval::BuildReportText(result, context);
    {
        std::ofstream file(reportTextPath.c_str(), std::ios::out | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "touch_eval: cannot write '" << reportTextPath << "'\n";
            return 2;
        }
        file << reportText;
    }
    if (!touch_eval::WriteReportJson(result, context, reportJsonPath, &error) ||
        !WriteTouchEventsJsonl(result, eventsPath, &error) ||
        (options.writeCsv && !WriteContactSignalCsv(result.contacts, csvPath, &error))) {
        std::cerr << "touch_eval: " << error << "\n";
        return 2;
    }

    std::cout << reportText;
    std::cout << "\noutputs:\n"
              << "  " << reportTextPath << "\n"
              << "  " << reportJsonPath << "\n"
              << "  " << eventsPath << "\n";
    if (options.writeCsv) std::cout << "  " << csvPath << "\n";
    if (reader.SkippedLines() > 0) {
        std::cout << "skipped " << reader.SkippedLines()
                  << " unparseable observation line(s)\n";
    }
    return 0;
}

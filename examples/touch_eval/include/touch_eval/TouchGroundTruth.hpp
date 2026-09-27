#pragma once

#include <string>
#include <vector>

namespace touch_eval {

/**
 * @file TouchGroundTruth.hpp
 * @brief The handwritten touch annotations and their machine readable form.
 *
 * Two source notations are supported.
 *
 * The legacy one is data/移动传球.label, typed by hand and mixing several
 * notations:
 *
 *   00:29S  (1次)                       mm:ss
 *   00:34S - 00:35 S (1次)              an interval (the touch process lasted
 *                                       more than one second)
 *   01:46S - 01:47S (2次) - 左右脚各1次  count plus a left/right hint
 *   00:38S - 0.39 (1次)                 typo for 00:38 - 00:39
 *   0.01S  (1次)                        dotted minutes:seconds (a typo for
 *                                       00:01, kept supported)
 *
 * The v2 one is a one-row-per-touch CSV:
 *
 *   index,type,t_start_sec,t_end_sec,count,foot,player,low_quality_data,note
 *   1,touch,1.00,1.00,1,,,,,
 *   2,touch,34.20,35.10,1,right,,yes,
 *   3,no_touch,12.00,15.00,0,,,,keeper picks the ball up
 *
 * `no_touch` rows are negative samples: the annotator confirms no touch
 * happened in that span, which is the only cheap way to measure precision.
 * Times are read off the `time=` burned into the rendered video, so 0.01 s
 * costs the annotator nothing extra.
 *
 * `low_quality_data` marks a touch whose frames are too degraded to judge
 * (a tiny player, motion blur on the ball). It is not a comment: the evaluator
 * scores the file twice, with and without those rows, because a detection the
 * annotator cannot verify is not evidence against the detector. `yes`, `true`,
 * `1`, `低质量` and `模糊` all set it; empty, `no`, `0` and `ok` clear it.
 *
 * Frames are optional but take precedence when present, so a row can be pinned
 * exactly without retyping its times:
 *
 *   # fps: 60
 *   index,type,t_start_sec,t_end_sec,t_start_frame,t_end_frame,count,foot,player,note
 *   4,touch,266.83,266.83,16010,16010,1,right,,confirmed by eye
 *
 * `frame` (a single column) sets both ends. Without a declared fps the frame
 * columns are reported and ignored rather than guessed at.
 *
 * Neither parser rewrites its source file: every event keeps the original line,
 * so a suspicious parse can be traced back (and inspected in the dumped JSON)
 * instead of being silently "fixed".
 */

/// One annotated touch, one annotated touch process, or one negative span.
struct GroundTruthEvent {
    /// 1-based event index in file order.
    int index = 0;
    /// 1-based line number in the source file.
    int sourceLine = 0;
    /// 1-based ordinal among the data rows of the file, 0 when unknown. It is
    /// what keeps a rewrite of the file in the annotator's own order instead of
    /// grouping every touch before the negative spans.
    int sourceRow = 0;
    /// Touch instant; for an interval annotation, the beginning.
    double startSec = 0.0;
    /// Touch instant of a point annotation (== startSec), or the end of an
    /// interval annotation.
    double endSec = 0.0;
    /// Frame numbers when the annotation states them (`t_start_frame` /
    /// `t_end_frame`, or `frame`), or when the file declares an fps and the
    /// seconds were converted. -1 = unknown. Frames win over seconds, so a row
    /// can be pinned to the exact frame without touching its times.
    int startFrame = -1;
    int endFrame = -1;
    /// How many touches the annotation counts: from "(N次)" in a .label, from
    /// the (single) row in a v2 CSV. Zero for a `no_touch` span.
    int count = 1;
    /// "left", "right", "both" or empty when the line says nothing about it.
    std::string footHint;
    /// Optional player label from the CSV `player` column.
    std::string playerHint;
    /// The annotator cannot judge this touch: the frames are too degraded (a
    /// player only a few pixels tall, the ball smeared by motion blur). Scored
    /// twice, with and without these rows, and never counted as detector error.
    bool lowQuality = false;
    /// Free text left on the line after the time and the count.
    std::string note;
    /// The line exactly as it was typed.
    std::string rawLine;
    /// Non-fatal remarks about this line, e.g. an interpreted typo.
    std::vector<std::string> warnings;

    bool IsInterval() const { return endSec > startSec; }
    double DurationSec() const { return endSec > startSec ? endSec - startSec : 0.0; }
};

struct GroundTruth {
    /// File the events were parsed from (or the source recorded in the JSON).
    std::string source;
    /// "label", "csv" or "json": decides the default point annotation window,
    /// because the notations were typed at different resolutions.
    std::string sourceKind = "label";
    /// Frame rate declared by `# fps: 60` in the annotations (or --fps on the
    /// command line). Needed to read frame based rows; 0 = unknown, in which
    /// case a frame column is reported and ignored.
    double fps = 0.0;
    /// Width of the window a point annotation stands for, centred on the
    /// instant it names. A legacy .label records "the touch happened in this
    /// second" (1.0 s, i.e. the old +-0.5 s tolerance); a v2 CSV written with
    /// 0.01 s times means "right about here" (0.5 s). A CSV converted from a
    /// .label keeps 1.0 s and says so in its header.
    double defaultPointWindowSec = 1.0;
    std::vector<GroundTruthEvent> events;
    /// Spans the annotator confirmed contain no touch (negative samples).
    std::vector<GroundTruthEvent> noTouch;
    /// Data rows the parser had to drop. A file that rewrites itself must refuse
    /// to do so while this is non-zero, or a typo would quietly delete a row.
    int skippedRows = 0;
    /// File level remarks: skipped lines and other global findings.
    std::vector<std::string> warnings;

    /// Sum of the per-event counts, i.e. the number of touches the annotation
    /// says happened.
    int TouchCount() const;
    /// Rows flagged as too degraded to judge, touches and negative spans alike.
    int LowQualityRows() const;
    /// Sum of the counts of the flagged touch rows.
    int LowQualityTouchCount() const;
    /// True when the annotations are usable at all.
    bool Empty() const { return events.empty(); }
};

/// Parse a .label file. Returns false and fills `error` when the file cannot be
/// read at all; unparseable lines are reported through `warnings` and skipped.
bool ParseTouchLabelFile(const std::string& path, GroundTruth* groundTruth,
                         std::string* error);

/// Parse label text that is already in memory. `source` is only used for
/// reporting.
GroundTruth ParseTouchLabelText(const std::string& text,
                                const std::string& source);

/// Parse a v2 one-row-per-touch CSV. Unusable rows are reported through
/// `warnings` and skipped; only an unreadable file fails. `fpsOverride` (from
/// --fps) wins over a `# fps:` hint in the file, and is what makes frame
/// columns readable when the file forgets to declare the rate.
bool ParseTouchCsvFile(const std::string& path, GroundTruth* groundTruth,
                       std::string* error, double fpsOverride = 0.0);

/// Parse v2 CSV text already in memory.
GroundTruth ParseTouchCsvText(const std::string& text,
                              const std::string& source,
                              double fpsOverride = 0.0);

/// Declare the frame rate of an already parsed file (--fps) and fill in the
/// frames of rows that only have seconds.
void ApplyFpsToGroundTruth(GroundTruth* groundTruth, double fps);

/// Write the annotations as a v2 CSV. One row per event, `count` preserved, and
/// the rows keep the order they were read in (a hand written file interleaves
/// touches and negative spans by time; regrouping them would only make it
/// harder to read).
bool WriteTouchCsv(const GroundTruth& groundTruth, const std::string& path,
                   std::string* error);

/// Rewrite a v2 CSV in place with sequential indices and the canonical column
/// set, keeping every other field exactly as the annotator wrote it. Hand
/// editing leaves duplicated or skipped indices; the evaluator ignores the
/// column, but a file that numbers itself is easier to talk about. The previous
/// content is kept next to it as `<path>.bak`. `rows` receives the row count.
bool RenumberTouchCsvFile(const std::string& path, int* rows, std::string* error);

/// Write a v2 CSV out of a legacy .label parse, expanding "(N次)" into that
/// many one-touch rows (same instant) and marking them in `note` so the
/// annotator knows which rows are still worth splitting by hand.
bool ConvertTouchLabelToCsv(const GroundTruth& labelGroundTruth,
                            const std::string& path, std::string* error);

/// Pick the parser from the file extension (.csv / .json / anything else =
/// legacy label).
bool ParseTouchAnnotationFile(const std::string& path, GroundTruth* groundTruth,
                              std::string* error, double fpsOverride = 0.0);

/// Serialize / deserialize the machine readable form (`--dump_gt`).
bool WriteGroundTruthJson(const GroundTruth& groundTruth, const std::string& path,
                          std::string* error);
bool ReadGroundTruthJson(const std::string& path, GroundTruth* groundTruth,
                         std::string* error);

/// "1:23.45" style helper used by the reports.
std::string FormatSeconds(double seconds);
/// "mm:ss.S" timestamp with a sign, used for the timing error lists.
std::string FormatSignedSeconds(double seconds);

} // namespace touch_eval

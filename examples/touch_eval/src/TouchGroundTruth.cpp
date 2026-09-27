#include <touch_eval/TouchGroundTruth.hpp>

#include <json/json.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

namespace touch_eval {
namespace {

bool IsSpace(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
           value == '\f' || value == '\v';
}

std::string Trim(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && IsSpace(text[begin])) ++begin;
    std::size_t end = text.size();
    while (end > begin && IsSpace(text[end - 1])) --end;
    return text.substr(begin, end - begin);
}

/// Next UTF-8 code point at `index`, advancing it. A malformed byte is
/// returned on its own so the scan always makes progress.
unsigned int DecodeUtf8(const std::string& text, std::size_t* index) {
    const unsigned char lead = static_cast<unsigned char>(text[*index]);
    std::size_t length = 1;
    unsigned int value = lead;
    if ((lead & 0xE0) == 0xC0) {
        length = 2;
        value = lead & 0x1Fu;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
        value = lead & 0x0Fu;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
        value = lead & 0x07u;
    }
    if (*index + length > text.size()) length = 1;
    for (std::size_t i = 1; i < length; ++i) {
        const unsigned char next = static_cast<unsigned char>(text[*index + i]);
        if ((next & 0xC0) != 0x80) {
            length = 1;
            break;
        }
        value = (value << 6) | (next & 0x3Fu);
    }
    *index += length;
    return value;
}

/// Separators that glue a trailing remark to the annotation, as code points:
/// working on bytes would cut a full width character in half and leave a
/// stray replacement character in the note.
bool IsNoteSeparator(unsigned int codePoint) {
    switch (codePoint) {
        case ' ': case '\t': case '-': case '~': case ',': case ':':
        case '.': case '_': case '/': case ')': case ']': case '|':
            return true;
        case 0xFF0C:  // ，
        case 0x3001:  // 、
        case 0xFF1A:  // ：
        case 0x3002:  // 。
        case 0xFF1B:  // ；
        case 0xFF09:  // ）
        case 0x2014:  // —
        case 0x2013:  // –
        case 0xFF5E:  // ～
        case 0x2212:  // −
        case 0x00B7:  // ·
            return true;
        default:
            return false;
    }
}

std::string TrimNote(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t cursor = begin;
        if (!IsNoteSeparator(DecodeUtf8(text, &cursor))) break;
        begin = cursor;
    }
    std::size_t end = text.size();
    while (end > begin && IsSpace(text[end - 1])) --end;
    return text.substr(begin, end - begin);
}

/// One timestamp found on a line.
struct TimeToken {
    std::size_t begin = 0;
    std::size_t end = 0;
    double seconds = 0.0;
    bool dotted = false;  // written with '.' instead of ':', e.g. "0.39"
    bool hasHours = false;
};

std::string FootHintOf(const std::string& text) {
    // "左右脚" contains "右脚" as a substring, so it has to be tested first.
    if (text.find("左右脚") != std::string::npos) return "both";
    if (text.find("右脚") != std::string::npos) return "right";
    if (text.find("左脚") != std::string::npos) return "left";
    return std::string();
}

bool ContainsDash(const std::string& text) {
    if (text.find('-') != std::string::npos) return true;
    if (text.find('~') != std::string::npos) return true;
    if (text.find("到") != std::string::npos) return true;
    // U+2014 em dash and U+FF5E full width tilde start with 0xE2 / 0xEF.
    if (text.find("\xE2\x80\x94") != std::string::npos) return true;
    if (text.find("\xEF\xBD\x9E") != std::string::npos) return true;
    return false;
}

void ParseLine(const std::string& rawLine, int lineNumber, int* eventIndex,
               GroundTruth* groundTruth) {
    std::string line = Trim(rawLine);
    if (line.empty() || line[0] == '#') return;

    // "(1次)" / "（2次）": the number of touches the annotation counts. The
    // multi-byte parentheses are alternatives, not a byte class: inside a
    // character class each UTF-8 byte would be a separate alternative and the
    // match could end on a partial character.
    static const std::regex countPattern(
        "(?:\\(|（)\\s*([0-9]+)\\s*次\\s*(?:\\)|）)");
    std::smatch countMatch;
    int count = 1;
    std::size_t countEnd = 0;
    const bool hasCount = std::regex_search(line, countMatch, countPattern);
    if (hasCount) {
        count = std::max(1, std::atoi(countMatch[1].str().c_str()));
        countEnd = static_cast<std::size_t>(countMatch.position() + countMatch.length());
    }

    // Timestamps: "mm:ss", "h:mm:ss" or the dotted typo "mm.ss".
    static const std::regex timePattern(
        "([0-9]{1,3})\\s*([:.：])\\s*([0-9]{1,2})"
        "(?:\\s*([:.：])\\s*([0-9]{1,2}))?");
    std::vector<TimeToken> tokens;
    for (std::sregex_iterator it(line.begin(), line.end(), timePattern), end;
         it != end; ++it) {
        const std::smatch& match = *it;
        TimeToken token;
        token.begin = static_cast<std::size_t>(match.position());
        token.end = token.begin + static_cast<std::size_t>(match.length());
        const int first = std::atoi(match[1].str().c_str());
        const int second = std::atoi(match[3].str().c_str());
        token.dotted = (match[2].str() == ".");
        if (match[5].matched) {
            token.hasHours = true;
            const int third = std::atoi(match[5].str().c_str());
            token.seconds = first * 3600.0 + second * 60.0 + third;
        } else {
            token.seconds = first * 60.0 + second;
        }
        tokens.push_back(token);
    }

    std::vector<std::string> warnings;
    if (tokens.empty()) {
        std::ostringstream message;
        message << "line " << lineNumber << " skipped, no timestamp: \"" << line << "\"";
        groundTruth->warnings.push_back(message.str());
        return;
    }

    GroundTruthEvent event;
    event.sourceLine = lineNumber;
    event.rawLine = rawLine;
    event.count = count;
    event.startSec = tokens[0].seconds;
    event.endSec = tokens[0].seconds;
    std::size_t lastTimeEnd = tokens[0].end;

    if (tokens[0].dotted) {
        std::ostringstream message;
        message << "line " << lineNumber << ": dotted time read as mm:ss";
        warnings.push_back(message.str());
    }

    if (tokens.size() > 1) {
        const std::string between =
            line.substr(tokens[0].end, tokens[1].begin - tokens[0].end);
        if (ContainsDash(between)) {
            event.endSec = tokens[1].seconds;
            lastTimeEnd = tokens[1].end;
            if (tokens[1].dotted) {
                std::ostringstream message;
                message << "line " << lineNumber
                        << ": dotted interval end read as mm:ss";
                warnings.push_back(message.str());
            }
            if (event.endSec < event.startSec) {
                std::swap(event.endSec, event.startSec);
                std::ostringstream message;
                message << "line " << lineNumber << ": interval end before start, swapped";
                warnings.push_back(message.str());
            }
            if (event.DurationSec() > 10.0) {
                std::ostringstream message;
                message << "line " << lineNumber << ": interval longer than 10 s ("
                        << event.DurationSec() << " s), please verify";
                warnings.push_back(message.str());
            }
        } else {
            std::ostringstream message;
            message << "line " << lineNumber << ": extra timestamp ignored: \""
                    << line.substr(tokens[1].begin, tokens[1].end - tokens[1].begin)
                    << "\"";
            warnings.push_back(message.str());
        }
    }

    const std::size_t noteBegin = std::max(lastTimeEnd, countEnd);
    if (noteBegin < line.size()) event.note = TrimNote(line.substr(noteBegin));
    event.footHint = FootHintOf(line);

    if (!hasCount) {
        std::ostringstream message;
        message << "line " << lineNumber << ": no (N次) count, assuming 1";
        warnings.push_back(message.str());
    }

    event.warnings = warnings;
    ++(*eventIndex);
    event.index = *eventIndex;
    groundTruth->events.push_back(event);
}

} // namespace

int GroundTruth::TouchCount() const {
    int total = 0;
    for (std::size_t i = 0; i < events.size(); ++i) total += events[i].count;
    return total;
}

int GroundTruth::LowQualityRows() const {
    int total = 0;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].lowQuality) ++total;
    }
    for (std::size_t i = 0; i < noTouch.size(); ++i) {
        if (noTouch[i].lowQuality) ++total;
    }
    return total;
}

int GroundTruth::LowQualityTouchCount() const {
    int total = 0;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].lowQuality) total += events[i].count;
    }
    return total;
}

GroundTruth ParseTouchLabelText(const std::string& text,
                                const std::string& source) {
    GroundTruth groundTruth;
    groundTruth.source = source;
    groundTruth.sourceKind = "label";
    // A handwritten line records "the touch happened somewhere in this second",
    // which is exactly a one second window, not a 0.00 s instant.
    groundTruth.defaultPointWindowSec = 1.0;

    std::istringstream stream(text);
    std::string line;
    int lineNumber = 0;
    int eventIndex = 0;
    while (std::getline(stream, line)) {
        ++lineNumber;
        if (lineNumber == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line = line.substr(3);
        }
        ParseLine(line, lineNumber, &eventIndex, &groundTruth);
    }
    return groundTruth;
}

bool ParseTouchLabelFile(const std::string& path, GroundTruth* groundTruth,
                         std::string* error) {
    if (!groundTruth) return false;
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        if (error) *error = "cannot open label file '" + path + "'";
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    *groundTruth = ParseTouchLabelText(buffer.str(), path);
    return true;
}

// ---------------------------------------------------------------------------
// v2: one row per touch
// ---------------------------------------------------------------------------

namespace {

/// Splits one CSV line. Double quotes group a field, `""` is a literal quote,
/// and both ASCII and full width commas separate fields.
std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool inQuotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char value = line[i];
        if (inQuotes) {
            if (value == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current.push_back('"');
                    ++i;
                } else {
                    inQuotes = false;
                }
            } else {
                current.push_back(value);
            }
            continue;
        }
        if (value == '"') {
            inQuotes = true;
        } else if (value == ',' || value == "\xEF\xBC\x8C"[0]) {
            // A full width comma is three bytes; treat the leading byte as a
            // separator only when the whole sequence is present.
            if (value == ',' ||
                line.compare(i, 3, "\xEF\xBC\x8C") == 0) {
                fields.push_back(Trim(current));
                current.clear();
                if (value != ',') i += 2;
            } else {
                current.push_back(value);
            }
        } else {
            current.push_back(value);
        }
    }
    fields.push_back(Trim(current));
    return fields;
}

std::string LowerAscii(const std::string& text) {
    std::string out = text;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (out[i] >= 'A' && out[i] <= 'Z') {
            out[i] = static_cast<char>(out[i] - 'A' + 'a');
        }
    }
    return out;
}

std::string StripBom(const std::string& line) {
    if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF) {
        return line.substr(3);
    }
    return line;
}

/// Column meaning of a CSV field.
enum class CsvColumn {
    Ignored = 0,
    Index,
    Type,
    Start,
    End,
    StartFrame,
    EndFrame,
    Frame,
    Count,
    Foot,
    Player,
    LowQuality,
    Note,
};

/// Column names that mark a touch as too degraded to judge. Both the column
/// header and the cell value are matched case-insensitively, and the Chinese
/// words the annotator is likely to type are accepted, so a hand written file
/// does not have to look up the exact spelling.
bool IsLowQualityColumnName(const std::string& key) {
    return key == "low_quality_data" || key == "low_quality" ||
           key == "lowquality" || key == "lowq" || key == "low" ||
           key == "quality" || key == "blur" || key == "blurry" ||
           key == "低质量" || key == "模糊" || key == "低质量数据";
}

/// True / false / "the annotator wrote something we do not know". The third
/// outcome is a warning rather than a silent no, because a mistyped flag would
/// quietly move a row from one score to the other.
enum class FlagValue { Unset = 0, Set, Unknown };

FlagValue ReadFlag(const std::string& text) {
    const std::string key = LowerAscii(Trim(text));
    if (key.empty() || key == "no" || key == "n" || key == "false" ||
        key == "0" || key == "-" || key == "ok" || key == "none" ||
        key == "正常" || key == "否" || key == "无") {
        return FlagValue::Unset;
    }
    if (key == "yes" || key == "y" || key == "true" || key == "1" ||
        key == "x" || key == "low" || key == "blur" || key == "blurry" ||
        key == "低" || key == "低质量" || key == "模糊" ||
        key == "低质量数据") {
        return FlagValue::Set;
    }
    return FlagValue::Unknown;
}

CsvColumn ClassifyColumn(const std::string& name) {
    const std::string key = LowerAscii(Trim(name));
    if (key == "index" || key == "idx" || key == "id" || key == "序号") {
        return CsvColumn::Index;
    }
    if (key == "type" || key == "kind" || key == "类型") return CsvColumn::Type;
    if (key == "t_start_sec" || key == "t_start" || key == "start_sec" ||
        key == "start" || key == "开始") {
        return CsvColumn::Start;
    }
    if (key == "t_end_sec" || key == "t_end" || key == "end_sec" ||
        key == "end" || key == "结束") {
        return CsvColumn::End;
    }
    // Frame columns: named only, so a headerless file keeps its old meaning.
    if (key == "t_start_frame" || key == "start_frame" ||
        key == "begin_frame" || key == "frame_start" || key == "开始帧") {
        return CsvColumn::StartFrame;
    }
    if (key == "t_end_frame" || key == "end_frame" || key == "frame_end" ||
        key == "结束帧") {
        return CsvColumn::EndFrame;
    }
    if (key == "frame" || key == "t_frame" || key == "帧") {
        return CsvColumn::Frame;
    }
    if (key == "count" || key == "n" || key == "times" || key == "次数") {
        return CsvColumn::Count;
    }
    if (key == "foot" || key == "side" || key == "脚" || key == "脚别") {
        return CsvColumn::Foot;
    }
    if (key == "player" || key == "player_id" || key == "球员") {
        return CsvColumn::Player;
    }
    if (IsLowQualityColumnName(key)) return CsvColumn::LowQuality;
    if (key == "note" || key == "remark" || key == "comment" || key == "备注") {
        return CsvColumn::Note;
    }
    return CsvColumn::Ignored;
}

/// Accepts "12.34", "12", "1:23.45", "00:01:02.5" and a trailing "s".
bool ParseCsvTime(const std::string& text, double* seconds) {
    std::string value = Trim(text);
    while (!value.empty() && (value[value.size() - 1] == 's' ||
                              value[value.size() - 1] == 'S')) {
        value.erase(value.size() - 1);
        value = Trim(value);
    }
    if (value.empty()) return false;

    std::vector<double> parts;
    std::string current;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const char character = value[i];
        if (character == ':' || character == "\xEF\xBC\x9A"[0]) {
            if (character != ':' && value.compare(i, 3, "\xEF\xBC\x9A") != 0) {
                return false;
            }
            if (character != ':') i += 2;
            parts.push_back(std::atof(current.c_str()));
            current.clear();
            continue;
        }
        if (!((character >= '0' && character <= '9') || character == '.' ||
              character == '+' || character == '-' || character == ' ')) {
            return false;
        }
        current.push_back(character);
    }
    parts.push_back(std::atof(current.c_str()));

    double total = 0.0;
    for (std::size_t i = 0; i < parts.size(); ++i) total = total * 60.0 + parts[i];
    *seconds = total;
    return true;
}

std::string NormalizeFootHint(const std::string& text) {
    const std::string key = LowerAscii(Trim(text));
    if (key.empty()) return std::string();
    if (key == "left" || key == "l" || text.find("左脚") != std::string::npos) {
        return "left";
    }
    if (key == "right" || key == "r" || text.find("右脚") != std::string::npos) {
        return "right";
    }
    if (key == "both" || text.find("左右脚") != std::string::npos) return "both";
    return std::string();
}

/// True for the `type` values that mean "no touch happened here".
bool IsNoTouchType(const std::string& text) {
    const std::string key = LowerAscii(Trim(text));
    return key == "no_touch" || key == "no-touch" || key == "notouch" ||
           key == "none" || key == "negative" || key == "no" ||
           key == "无触球" || key == "无" || key == "空";
}

bool IsTouchType(const std::string& text) {
    const std::string key = LowerAscii(Trim(text));
    return key.empty() || key == "touch" || key == "t" || key == "yes" ||
           key == "触球" || key == "有";
}

std::string CsvEscape(const std::string& text) {
    if (text.find_first_of(",\"\n") == std::string::npos) return text;
    std::string out = "\"";
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '"') out += "\"\"";
        else out.push_back(text[i]);
    }
    out += "\"";
    return out;
}

std::string FormatCsvSeconds(double seconds) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f", seconds);
    return std::string(buffer);
}

/// Reads `# point_window_sec: 1.0` / `# fps: 60` style hints out of the comment
/// header, so a file declares how it should be read instead of the reader
/// guessing.
bool ParseNumberHint(const std::string& comment, const std::string& key,
                     double* value) {
    if (comment.empty() || comment[0] != '#') return false;
    const std::string body = LowerAscii(comment);
    const std::size_t position = body.find(key);
    if (position == std::string::npos) return false;
    std::string rest = comment.substr(position + key.size());
    // A frame hint is "fps", not "fpsomething".
    if (!rest.empty() && ((rest[0] >= 'a' && rest[0] <= 'z') ||
                          (rest[0] >= 'A' && rest[0] <= 'Z'))) {
        return false;
    }
    // Accept a trailing unit, then any of ": = \t".
    const std::string restLower = LowerAscii(rest);
    if (restLower.compare(0, 3, "sec") == 0) rest = rest.substr(3);
    std::size_t begin = 0;
    while (begin < rest.size() && !(rest[begin] >= '0' && rest[begin] <= '9')) {
        ++begin;
    }
    if (begin >= rest.size()) return false;
    *value = std::atof(rest.substr(begin).c_str());
    return *value > 0.0;
}

} // namespace

void ApplyFpsToGroundTruth(GroundTruth* groundTruth, double fps) {
    if (!groundTruth || fps <= 0.0) return;
    groundTruth->fps = fps;
    const auto fill = [fps](std::vector<GroundTruthEvent>* events) {
        for (std::size_t i = 0; i < events->size(); ++i) {
            GroundTruthEvent& event = (*events)[i];
            if (event.startFrame < 0) {
                event.startFrame = static_cast<int>(
                    std::lround(event.startSec * fps));
            }
            if (event.endFrame < 0) {
                event.endFrame = static_cast<int>(
                    std::lround(event.endSec * fps));
            }
        }
    };
    fill(&groundTruth->events);
    fill(&groundTruth->noTouch);
}

GroundTruth ParseTouchCsvText(const std::string& text,
                              const std::string& source,
                              double fpsOverride) {
    GroundTruth groundTruth;
    groundTruth.source = source;
    groundTruth.sourceKind = "csv";
    // Default: a CSV time is read off the rendered overlay, i.e. a 0.01 s
    // instant, which is worth half a second of window. A file converted from a
    // second-resolution .label says so in its header and keeps the full second.
    groundTruth.defaultPointWindowSec = 0.5;
    if (fpsOverride > 0.0) groundTruth.fps = fpsOverride;

    std::vector<CsvColumn> columns;
    bool haveHeader = false;
    std::istringstream stream(text);
    std::string line;
    int lineNumber = 0;
    int fallbackIndex = 0;
    int dataRow = 0;
    std::set<int> usedIndices;

    // The `index` column is what the annotator sees in the file, and the report
    // talks about "gt#N", so honoring it is what keeps the two in step - a
    // negative span in the middle of the file keeps its number instead of
    // pushing every later touch down by one. A repeated index (hand editing
    // happens) falls back to the running number and says so.
    const auto assignIndex = [&](int statedIndex, int line,
                                 std::vector<std::string>* warnings) {
        if (statedIndex > 0 && usedIndices.insert(statedIndex).second) {
            return statedIndex;
        }
        int candidate = ++fallbackIndex;
        while (!usedIndices.insert(candidate).second) candidate = ++fallbackIndex;
        if (statedIndex > 0) {
            std::ostringstream message;
            message << "line " << line << ": index " << statedIndex
                    << " is already used, renumbered to " << candidate;
            warnings->push_back(message.str());
        }
        return candidate;
    };
    while (std::getline(stream, line)) {
        ++lineNumber;
        if (lineNumber == 1) line = StripBom(line);
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            double windowSec = 0.0;
            if (ParseNumberHint(trimmed, "point_window", &windowSec)) {
                groundTruth.defaultPointWindowSec = windowSec;
            }
            double fps = 0.0;
            // --fps outranks the file's own hint.
            if (fpsOverride <= 0.0 && ParseNumberHint(trimmed, "fps", &fps)) {
                groundTruth.fps = fps;
            }
            continue;
        }

        const std::vector<std::string> fields = SplitCsvLine(line);
        if (!haveHeader) {
            columns.clear();
            bool headerLooksRight = false;
            for (std::size_t i = 0; i < fields.size(); ++i) {
                const CsvColumn column = ClassifyColumn(fields[i]);
                columns.push_back(column);
                if (column == CsvColumn::Start) headerLooksRight = true;
            }
            if (headerLooksRight) {
                haveHeader = true;
                continue;
            }
            // No header: assume the documented column order of the original
            // four-plus-four layout. The low quality flag is deliberately left
            // out: reading an old note column as a flag would silently move a
            // row between the two scores.
            columns.clear();
            columns.push_back(CsvColumn::Index);
            columns.push_back(CsvColumn::Type);
            columns.push_back(CsvColumn::Start);
            columns.push_back(CsvColumn::End);
            columns.push_back(CsvColumn::Count);
            columns.push_back(CsvColumn::Foot);
            columns.push_back(CsvColumn::Player);
            columns.push_back(CsvColumn::Note);
            haveHeader = true;
        }

        GroundTruthEvent event;
        event.sourceLine = lineNumber;
        event.sourceRow = ++dataRow;
        event.rawLine = trimmed;
        bool hasStart = false;
        std::string typeText;
        int count = -1;
        int statedIndex = -1;
        int startFrame = -1;
        int endFrame = -1;
        int singleFrame = -1;

        for (std::size_t i = 0; i < fields.size() && i < columns.size(); ++i) {
            if (columns[i] == CsvColumn::Ignored || fields[i].empty()) continue;
            switch (columns[i]) {
                case CsvColumn::Type:
                    typeText = fields[i];
                    break;
                case CsvColumn::StartFrame:
                    startFrame = std::atoi(fields[i].c_str());
                    break;
                case CsvColumn::EndFrame:
                    endFrame = std::atoi(fields[i].c_str());
                    break;
                case CsvColumn::Frame:
                    singleFrame = std::atoi(fields[i].c_str());
                    break;
                case CsvColumn::Start: {
                    double value = 0.0;
                    if (!ParseCsvTime(fields[i], &value)) {
                        std::ostringstream message;
                        message << "line " << lineNumber
                                << ": cannot read start time \"" << fields[i]
                                << "\", row skipped";
                        groundTruth.warnings.push_back(message.str());
                        hasStart = false;
                        count = -2;
                        ++groundTruth.skippedRows;
                        break;
                    }
                    event.startSec = value;
                    hasStart = true;
                    break;
                }
                case CsvColumn::End: {
                    double value = 0.0;
                    if (ParseCsvTime(fields[i], &value)) event.endSec = value;
                    break;
                }
                case CsvColumn::Count:
                    count = std::atoi(fields[i].c_str());
                    break;
                case CsvColumn::Foot:
                    event.footHint = NormalizeFootHint(fields[i]);
                    break;
                case CsvColumn::Player:
                    event.playerHint = Trim(fields[i]);
                    break;
                case CsvColumn::LowQuality: {
                    const FlagValue flag = ReadFlag(fields[i]);
                    event.lowQuality = flag == FlagValue::Set;
                    if (flag == FlagValue::Unknown) {
                        std::ostringstream message;
                        message << "line " << lineNumber
                                << ": cannot read low_quality_data \""
                                << fields[i] << "\", read as not low quality";
                        event.warnings.push_back(message.str());
                    }
                    break;
                }
                case CsvColumn::Note:
                    event.note = Trim(fields[i]);
                    break;
                case CsvColumn::Index:
                    statedIndex = std::atoi(fields[i].c_str());
                    break;
                case CsvColumn::Ignored:
                    break;
            }
        }
        if (count == -2) continue;  // start time was unusable

        // Frames win over seconds: a row pinned by frame number is exact, and
        // it may well leave the seconds columns empty.
        const bool hasFrames = singleFrame >= 0 || startFrame >= 0;
        if (hasFrames) {
            if (groundTruth.fps <= 0.0) {
                std::ostringstream message;
                message << "line " << lineNumber
                        << ": frame given but the file declares no fps"
                           " (# fps: 60 or --fps), frames ignored";
                event.warnings.push_back(message.str());
            } else {
                int first = singleFrame >= 0 ? singleFrame : startFrame;
                int last = singleFrame >= 0 ? singleFrame : endFrame;
                if (last < 0) last = first;
                if (last < first) std::swap(last, first);
                event.startFrame = first;
                event.endFrame = last;
                event.startSec = static_cast<double>(first) / groundTruth.fps;
                event.endSec = static_cast<double>(last) / groundTruth.fps;
                hasStart = true;
            }
        }
        if (!hasStart) {
            std::ostringstream message;
            message << "line " << lineNumber
                    << ": no start time, row skipped: \"" << trimmed << "\"";
            groundTruth.warnings.push_back(message.str());
            ++groundTruth.skippedRows;
            continue;
        }
        if (event.endSec <= 0.0) event.endSec = event.startSec;
        // With a known fps every row can carry its frames, even the ones the
        // annotator wrote in seconds: the report and the review video then talk
        // about frames without recomputing anything.
        if (groundTruth.fps > 0.0 && event.startFrame < 0) {
            event.startFrame = static_cast<int>(
                std::lround(event.startSec * groundTruth.fps));
            event.endFrame = static_cast<int>(
                std::lround(event.endSec * groundTruth.fps));
        }
        if (event.endSec < event.startSec) {
            std::swap(event.endSec, event.startSec);
            std::ostringstream message;
            message << "line " << lineNumber
                    << ": end before start, swapped";
            event.warnings.push_back(message.str());
        }

        if (IsNoTouchType(typeText)) {
            event.count = 0;
            event.index = assignIndex(statedIndex, lineNumber, &event.warnings);
            groundTruth.noTouch.push_back(event);
            continue;
        }
        if (!IsTouchType(typeText)) {
            std::ostringstream message;
            message << "line " << lineNumber << ": unknown type \"" << typeText
                    << "\", read as a touch";
            event.warnings.push_back(message.str());
        }
        event.count = count >= 1 ? count : 1;
        if (count == 0) {
            std::ostringstream message;
            message << "line " << lineNumber
                    << ": touch row with count 0, assuming 1";
            event.warnings.push_back(message.str());
        }
        event.index = assignIndex(statedIndex, lineNumber, &event.warnings);
        groundTruth.events.push_back(event);
    }
    return groundTruth;
}

bool ParseTouchCsvFile(const std::string& path, GroundTruth* groundTruth,
                       std::string* error, double fpsOverride) {
    if (!groundTruth) return false;
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        if (error) *error = "cannot open touch csv '" + path + "'";
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    *groundTruth = ParseTouchCsvText(buffer.str(), path, fpsOverride);
    return true;
}

bool WriteTouchCsv(const GroundTruth& groundTruth, const std::string& path,
                   std::string* error) {
    std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        if (error) *error = "cannot write touch csv '" + path + "'";
        return false;
    }
    file << "# touch annotations v2 - one row per touch\n";
    file << "# type: touch | no_touch (no_touch = confirmed negative span)\n";
    file << "# t_end == t_start means an instant; times are seconds\n";
    // The window a point annotation stands for is a property of how the times
    // were written (whole seconds vs 0.01 s), so it travels with the file.
    file << "# point_window_sec: " << FormatCsvSeconds(
                groundTruth.defaultPointWindowSec > 0.0
                    ? groundTruth.defaultPointWindowSec : 0.5) << "\n";
    const bool withFrames = groundTruth.fps > 0.0;
    if (withFrames) {
        file << "# fps: " << FormatCsvSeconds(groundTruth.fps) << "\n";
        file << "# optional: fill t_start_frame / t_end_frame (or frame) to "
                "state a\n"
                "# touch by frame number; frames override the seconds above\n";
    }
    file << "index,type,t_start_sec,t_end_sec";
    if (withFrames) file << ",t_start_frame,t_end_frame";
    file << ",count,foot,player,low_quality_data,note\n";

    // Frames travel with the file once an fps is known, so a row can be pinned
    // exactly later without recomputing anything.
    const auto frameOf = [&groundTruth](int explicitFrame, double seconds) {
        if (explicitFrame >= 0) return explicitFrame;
        if (groundTruth.fps <= 0.0) return -1;
        return static_cast<int>(std::lround(seconds * groundTruth.fps));
    };

    // The annotator's own row order: touches and negative spans are interleaved
    // by time in a hand written file. `sourceRow` remembers where each row came
    // from; 0 means the file was built in memory, and then the two groups are
    // written one after the other as before.
    struct Row {
        int sourceRow = 0;
        bool noTouch = false;
        int index = 0;
    };
    std::vector<Row> rows;
    bool ordered = true;
    for (std::size_t i = 0; i < groundTruth.events.size(); ++i) {
        Row row;
        row.sourceRow = groundTruth.events[i].sourceRow;
        row.index = static_cast<int>(i);
        if (row.sourceRow <= 0) ordered = false;
        rows.push_back(row);
    }
    for (std::size_t i = 0; i < groundTruth.noTouch.size(); ++i) {
        Row row;
        row.sourceRow = groundTruth.noTouch[i].sourceRow;
        row.noTouch = true;
        row.index = static_cast<int>(i);
        if (row.sourceRow <= 0) ordered = false;
        rows.push_back(row);
    }
    if (ordered) {
        std::stable_sort(rows.begin(), rows.end(),
                         [](const Row& lhs, const Row& rhs) {
                             return lhs.sourceRow < rhs.sourceRow;
                         });
    }

    const auto writeRow = [&](std::ostream& out, int index,
                              const GroundTruthEvent& event, bool noTouch) {
        out << index << ',' << (noTouch ? "no_touch" : "touch") << ','
            << FormatCsvSeconds(event.startSec) << ','
            << FormatCsvSeconds(event.endSec);
        if (withFrames) {
            out << ',' << frameOf(event.startFrame, event.startSec) << ','
                << frameOf(event.endFrame, event.endSec);
        }
        out << ',' << (noTouch ? 0 : event.count) << ',' << event.footHint << ','
            << CsvEscape(event.playerHint) << ','
            << (event.lowQuality ? "yes" : "") << ',' << CsvEscape(event.note)
            << '\n';
    };

    int row = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Row& entry = rows[i];
        if (entry.noTouch) {
            writeRow(file, ++row, groundTruth.noTouch[entry.index], true);
        } else {
            writeRow(file, ++row, groundTruth.events[entry.index], false);
        }
    }
    return true;
}

bool RenumberTouchCsvFile(const std::string& path, int* rows,
                          std::string* error) {
    GroundTruth groundTruth;
    if (!ParseTouchCsvFile(path, &groundTruth, error)) return false;
    if (groundTruth.Empty()) {
        if (error) *error = "'" + path + "' has no usable row";
        return false;
    }
    // A row the parser could not read would be written back missing, which is
    // the one way this command can destroy work. Refuse instead, and point at
    // the warnings that explain which line is at fault.
    if (groundTruth.skippedRows > 0) {
        std::ostringstream message;
        message << "'" << path << "' has " << groundTruth.skippedRows
                << " row(s) that cannot be read, refusing to rewrite it";
        if (!groundTruth.warnings.empty()) {
            message << " (" << groundTruth.warnings.front() << ")";
        }
        if (error) *error = message.str();
        return false;
    }

    // Keep a copy first: this rewrites the annotator's own file, and a `.bak`
    // is cheaper than an argument about what the previous content was.
    const std::string backup = path + ".bak";
    {
        std::ifstream source(path.c_str(), std::ios::in | std::ios::binary);
        std::ofstream target(backup.c_str(), std::ios::out | std::ios::trunc |
                                                 std::ios::binary);
        if (!source.is_open() || !target.is_open()) {
            if (error) *error = "cannot copy '" + path + "' to '" + backup + "'";
            return false;
        }
        target << source.rdbuf();
        if (!target.good()) {
            if (error) *error = "cannot write '" + backup + "'";
            return false;
        }
    }

    const std::string temporary = path + ".tmp";
    if (!WriteTouchCsv(groundTruth, temporary, error)) return false;
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        if (error) *error = "cannot replace '" + path + "'";
        return false;
    }
    if (rows) {
        *rows = static_cast<int>(groundTruth.events.size() +
                                 groundTruth.noTouch.size());
    }
    return true;
}

bool ConvertTouchLabelToCsv(const GroundTruth& labelGroundTruth,
                            const std::string& path, std::string* error) {
    GroundTruth expanded;
    expanded.source = labelGroundTruth.source;
    expanded.sourceKind = "csv";
    // The times are still whole seconds: only the notation changed, not the
    // resolution, and the written header keeps saying so.
    expanded.defaultPointWindowSec = labelGroundTruth.defaultPointWindowSec > 0.0
        ? labelGroundTruth.defaultPointWindowSec
        : 1.0;
    // Keep the frames if the caller already knows the frame rate (a .label does
    // not, so the converted file simply has no frame columns).
    expanded.fps = labelGroundTruth.fps;
    expanded.warnings = labelGroundTruth.warnings;
    for (std::size_t i = 0; i < labelGroundTruth.events.size(); ++i) {
        const GroundTruthEvent& source = labelGroundTruth.events[i];
        for (int copy = 0; copy < std::max(1, source.count); ++copy) {
            GroundTruthEvent event = source;
            event.count = 1;
            if (source.count > 1) {
                std::ostringstream note;
                note << "from label line " << source.sourceLine << " ("
                     << source.count << "次)"
                     << (source.note.empty() ? "" : " - ") << source.note
                     << " - SPLIT ME: write the real time of each touch";
                event.note = note.str();
            }
            expanded.events.push_back(event);
        }
    }
    expanded.noTouch = labelGroundTruth.noTouch;
    for (std::size_t i = 0; i < expanded.events.size(); ++i) {
        expanded.events[i].index = static_cast<int>(i) + 1;
    }
    return WriteTouchCsv(expanded, path, error);
}

bool ParseTouchAnnotationFile(const std::string& path, GroundTruth* groundTruth,
                              std::string* error, double fpsOverride) {
    const std::size_t dot = path.find_last_of('.');
    const std::string extension =
        dot == std::string::npos ? std::string() : LowerAscii(path.substr(dot));
    if (extension == ".csv" || extension == ".tsv") {
        return ParseTouchCsvFile(path, groundTruth, error, fpsOverride);
    }
    if (extension == ".json") {
        if (!ReadGroundTruthJson(path, groundTruth, error)) return false;
        ApplyFpsToGroundTruth(groundTruth, fpsOverride);
        return true;
    }
    if (!ParseTouchLabelFile(path, groundTruth, error)) return false;
    ApplyFpsToGroundTruth(groundTruth, fpsOverride);
    return true;
}

bool WriteGroundTruthJson(const GroundTruth& groundTruth, const std::string& path,
                          std::string* error) {
    Json::Value root;
    root["schema_version"] = 2;
    root["source"] = groundTruth.source;
    root["source_kind"] = groundTruth.sourceKind;
    root["default_point_window_sec"] = groundTruth.defaultPointWindowSec;
    root["fps"] = groundTruth.fps;
    root["generated_by"] = "touch_eval --dump_gt";
    root["totals"]["events"] = static_cast<Json::UInt>(groundTruth.events.size());
    root["totals"]["touches"] = groundTruth.TouchCount();
    root["totals"]["no_touch_spans"] =
        static_cast<Json::UInt>(groundTruth.noTouch.size());
    root["totals"]["low_quality_rows"] = groundTruth.LowQualityRows();
    root["totals"]["low_quality_touches"] = groundTruth.LowQualityTouchCount();
    root["totals"]["skipped_rows"] = groundTruth.skippedRows;
    root["warnings"] = Json::arrayValue;
    for (std::size_t i = 0; i < groundTruth.warnings.size(); ++i) {
        root["warnings"].append(groundTruth.warnings[i]);
    }
    const auto fillEvent = [](const GroundTruthEvent& event) {
        Json::Value item;
        item["index"] = event.index;
        item["source_line"] = event.sourceLine;
        item["start_sec"] = event.startSec;
        item["end_sec"] = event.endSec;
        if (event.startFrame >= 0) item["start_frame"] = event.startFrame;
        if (event.endFrame >= 0) item["end_frame"] = event.endFrame;
        item["is_interval"] = event.IsInterval();
        item["count"] = event.count;
        item["foot_hint"] = event.footHint;
        item["player_hint"] = event.playerHint;
        item["low_quality"] = event.lowQuality;
        item["source_row"] = event.sourceRow;
        item["note"] = event.note;
        item["raw_line"] = event.rawLine;
        item["warnings"] = Json::arrayValue;
        for (std::size_t w = 0; w < event.warnings.size(); ++w) {
            item["warnings"].append(event.warnings[w]);
        }
        return item;
    };
    root["events"] = Json::arrayValue;
    for (std::size_t i = 0; i < groundTruth.events.size(); ++i) {
        root["events"].append(fillEvent(groundTruth.events[i]));
    }
    root["no_touch"] = Json::arrayValue;
    for (std::size_t i = 0; i < groundTruth.noTouch.size(); ++i) {
        root["no_touch"].append(fillEvent(groundTruth.noTouch[i]));
    }

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    // Keep the Chinese notes and the original annotation lines readable: this
    // file exists to be inspected when a parse is questioned.
    builder["emitUTF8"] = true;
    std::ofstream file(path.c_str(), std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        if (error) *error = "cannot write ground truth '" + path + "'";
        return false;
    }
    file << Json::writeString(builder, root) << '\n';
    return true;
}

bool ReadGroundTruthJson(const std::string& path, GroundTruth* groundTruth,
                         std::string* error) {
    if (!groundTruth) return false;
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        if (error) *error = "cannot open ground truth json '" + path + "'";
        return false;
    }
    Json::Value root;
    try {
        file >> root;
    } catch (const std::exception& exception) {
        if (error) {
            *error = std::string("cannot parse ground truth json: ") +
                     exception.what();
        }
        return false;
    }
    if (!root.isObject() || !root["events"].isArray()) {
        if (error) *error = "ground truth json has no 'events' array";
        return false;
    }

    GroundTruth parsed;
    parsed.source = root.get("source", std::string()).asString();
    parsed.sourceKind = root.get("source_kind", std::string("json")).asString();
    parsed.defaultPointWindowSec =
        root.get("default_point_window_sec", 1.0).asDouble();
    parsed.fps = root.get("fps", 0.0).asDouble();
    const Json::Value& warnings = root["warnings"];
    for (Json::ArrayIndex i = 0; i < warnings.size(); ++i) {
        parsed.warnings.push_back(warnings[i].asString());
    }
    const auto readEvent = [](const Json::Value& item, int fallbackIndex) {
        GroundTruthEvent event;
        event.index = item.get("index", fallbackIndex).asInt();
        event.sourceLine = item.get("source_line", 0).asInt();
        event.startSec = item.get("start_sec", 0.0).asDouble();
        event.endSec = item.get("end_sec", event.startSec).asDouble();
        event.count = item.get("count", 1).asInt();
        if (item.isMember("start_frame")) {
            event.startFrame = item["start_frame"].asInt();
        }
        if (item.isMember("end_frame")) {
            event.endFrame = item["end_frame"].asInt();
        }
        event.footHint = item.get("foot_hint", std::string()).asString();
        event.playerHint = item.get("player_hint", std::string()).asString();
        event.lowQuality = item.get("low_quality", false).asBool();
        event.sourceRow = item.get("source_row", 0).asInt();
        event.note = item.get("note", std::string()).asString();
        event.rawLine = item.get("raw_line", std::string()).asString();
        const Json::Value& eventWarnings = item["warnings"];
        for (Json::ArrayIndex w = 0; w < eventWarnings.size(); ++w) {
            event.warnings.push_back(eventWarnings[w].asString());
        }
        return event;
    };
    const Json::Value& events = root["events"];
    for (Json::ArrayIndex i = 0; i < events.size(); ++i) {
        parsed.events.push_back(readEvent(events[i], static_cast<int>(i) + 1));
    }
    const Json::Value& noTouch = root["no_touch"];
    for (Json::ArrayIndex i = 0; i < noTouch.size(); ++i) {
        parsed.noTouch.push_back(readEvent(noTouch[i], static_cast<int>(i) + 1));
    }
    *groundTruth = parsed;
    return true;
}

std::string FormatSeconds(double seconds) {
    const bool negative = seconds < 0.0;
    double value = negative ? -seconds : seconds;
    const int minutes = static_cast<int>(value / 60.0);
    value -= minutes * 60.0;
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%s%d:%05.2f", negative ? "-" : "",
                  minutes, value);
    return std::string(buffer);
}

std::string FormatSignedSeconds(double seconds) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%+.2f", seconds);
    return std::string(buffer);
}

} // namespace touch_eval

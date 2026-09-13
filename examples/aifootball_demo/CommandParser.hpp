#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace app {

/**
 * @brief Python-argparse-style command-line parser.
 *
 * Features:
 *   - `--long_name value` / `--long-name value` / `--long_name=value`
 *   - `-s value` (short form)
 *   - Hyphens and underscores are interchangeable (--stop-frame == --stop_frame)
 *   - Boolean flags: --verbose (no value)
 *   - Positional arguments
 *   - Required option validation
 *   - Default values
 *
 * Usage:
 *   CommandParser parser(argc, argv);
 *   parser.AddArgument("--config", "-c", "Path to config.yaml", true);
 *   parser.AddArgument("--stop_frame", "-s", "Stop after N frames", false, "0");
 *   parser.AddArgument("--verbose", "-v", "Enable debug logging", CommandParser::FLAG);
 *   if (!parser.Parse()) { parser.PrintHelp(); return -1; }
 *   auto cfg = parser.Get("config");
 *   int stop = parser.GetInt("stop_frame");
 */
class CommandParser {
public:
    /// Special default value to mark an option as a boolean flag.
    static const std::string& FlagMarker() {
        static const std::string s = "\x01FLAG";
        return s;
    }

    CommandParser(int argc, char* argv[]) {
        m_progName = argv[0];
        for (int i = 1; i < argc; ++i) m_args.push_back(argv[i]);
    }

    // -----------------------------------------------------------------------
    // Registration API (Python argparse style)
    // -----------------------------------------------------------------------

    /**
     * @brief Register an option.
     * @param longName   e.g. "--video_path" or "video_path" (leading -- optional)
     * @param shortName  e.g. "-v" or 'v' (leading - optional), 0 = none
     * @param help       Description text
     * @param required   If true, Parse() fails when missing
     * @param defaultVal Default value as string; use FLAG for boolean flags
     */
    void AddArgument(const std::string& longName, const std::string& shortName = "",
                     const std::string& help = "", bool required = false,
                     const std::string& defaultVal = "") {
        OptionDef def;
        def.name = StripDashes(longName);
        def.shortName = StripDashes(shortName);
        def.help = help;
        def.required = required;
        def.isFlag = (defaultVal == FlagMarker());
        def.defaultValue = def.isFlag ? "false" : defaultVal;
        def.isPositional = false;
        m_defs.push_back(def);
    }

    /** Overload: char short name. */
    void AddArgument(const std::string& longName, char shortChar,
                     const std::string& help = "", bool required = false,
                     const std::string& defaultVal = "") {
        std::string s(1, shortChar);
        AddArgument(longName, s, help, required, defaultVal);
    }

    /** @brief Register a positional argument. */
    void AddPositional(const std::string& name, const std::string& help = "",
                       bool required = false) {
        OptionDef def;
        def.name = StripDashes(name);
        def.help = help;
        def.required = required;
        def.isFlag = false;
        def.isPositional = true;
        m_defs.push_back(def);
    }

    // -----------------------------------------------------------------------
    // Parse
    // -----------------------------------------------------------------------

    bool Parse() {
        size_t positionalIdx = 0;

        for (size_t i = 0; i < m_args.size(); ++i) {
            const std::string& arg = m_args[i];

            // Built-in help.
            if (arg == "--help" || arg == "-h") {
                m_values["help"] = "true";
                continue;
            }

            // Long option: --key=value or --key value or --flag
            if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
                std::string raw = arg.substr(2);
                std::string key, val;
                bool hasEq = false;

                auto eqPos = raw.find('=');
                if (eqPos != std::string::npos) {
                    key = Normalize(raw.substr(0, eqPos));
                    val = raw.substr(eqPos + 1);
                    hasEq = true;
                } else {
                    key = Normalize(raw);
                }

                const OptionDef* def = FindDef(key);

                if (!hasEq) {
                    if (def && def->isFlag) {
                        m_values[key] = "true";
                        continue;
                    }
                    // Consume next arg as value.
                    if (i + 1 < m_args.size()) {
                        val = m_args[++i];
                    } else {
                        m_error = "Option --" + key + " requires a value";
                        return false;
                    }
                }
                m_values[key] = val;
                continue;
            }

            // Short option: -k value or -k (flag) or -abc (multi-flag)
            if (arg.size() >= 2 && arg[0] == '-' && arg[1] != '-') {
                // Handle bundled flags like -vq
                if (arg.size() > 2 && AllShortFlags(arg.substr(1))) {
                    for (size_t c = 1; c < arg.size(); ++c) {
                        const OptionDef* def = FindDefByShort(std::string(1, arg[c]));
                        std::string key = def ? Normalize(def->name) : std::string(1, arg[c]);
                        m_values[key] = "true";
                    }
                    continue;
                }

                char shortKey = arg[1];
                std::string shortStr(1, shortKey);
                const OptionDef* def = FindDefByShort(shortStr);
                std::string key = def ? Normalize(def->name) : shortStr;

                if (def && def->isFlag) {
                    m_values[key] = "true";
                    continue;
                }
                if (i + 1 < m_args.size()) {
                    m_values[key] = m_args[++i];
                } else {
                    m_error = std::string("Option -") + shortKey + " requires a value";
                    return false;
                }
                continue;
            }

            // Positional argument.
            {
                const OptionDef* posDef = GetPositionalAt(positionalIdx);
                std::string key = posDef ? Normalize(posDef->name)
                                         : ("arg" + std::to_string(positionalIdx));
                m_values[key] = arg;
                positionalIdx++;
            }
        }

        // Skip validation if help was requested.
        if (IsFlagSet("help")) return true;

        // Validate required options.
        for (const auto& def : m_defs) {
            if (def.required && m_values.find(Normalize(def.name)) == m_values.end()) {
                m_error = "Required option --" + def.name + " is missing";
                return false;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------------
    // Accessors (key can use either - or _, auto-normalized)
    // -----------------------------------------------------------------------

    std::string Get(const std::string& key, const std::string& defaultVal = "") const {
        std::string nk = Normalize(key);
        auto it = m_values.find(nk);
        if (it != m_values.end()) return it->second;
        const OptionDef* def = FindDef(nk);
        if (def && !def->defaultValue.empty()) return def->defaultValue;
        return defaultVal;
    }

    int GetInt(const std::string& key, int defaultVal = 0) const {
        std::string nk = Normalize(key);
        auto it = m_values.find(nk);
        if (it == m_values.end()) {
            const OptionDef* def = FindDef(nk);
            if (def && !def->defaultValue.empty()) return std::atoi(def->defaultValue.c_str());
            return defaultVal;
        }
        return std::atoi(it->second.c_str());
    }

    float GetFloat(const std::string& key, float defaultVal = 0.0f) const {
        std::string nk = Normalize(key);
        auto it = m_values.find(nk);
        if (it == m_values.end()) {
            const OptionDef* def = FindDef(nk);
            if (def && !def->defaultValue.empty())
                return static_cast<float>(std::atof(def->defaultValue.c_str()));
            return defaultVal;
        }
        return static_cast<float>(std::atof(it->second.c_str()));
    }

    bool Has(const std::string& key) const {
        return m_values.find(Normalize(key)) != m_values.end();
    }

    bool IsFlagSet(const std::string& key) const {
        return Get(key, "false") == "true";
    }

    const std::string& Error() const { return m_error; }
    bool IsHelpRequested() const { return IsFlagSet("help"); }

    // -----------------------------------------------------------------------
    // Help output (Python argparse style)
    // -----------------------------------------------------------------------

    void PrintHelp() const {
        fprintf(stderr, "usage: %s", m_progName.c_str());

        // Positional args in usage line.
        for (const auto& def : m_defs) {
            if (def.isPositional) {
                if (def.required) fprintf(stderr, " %s", def.name.c_str());
                else              fprintf(stderr, " [%s]", def.name.c_str());
            }
        }
        fprintf(stderr, " [options]\n\n");

        // Positional arguments section.
        bool hasPositional = false;
        for (const auto& def : m_defs) {
            if (def.isPositional) {
                if (!hasPositional) {
                    fprintf(stderr, "positional arguments:\n");
                    hasPositional = true;
                }
                fprintf(stderr, "  %-24s %s", def.name.c_str(), def.help.c_str());
                if (def.required) fprintf(stderr, " (required)");
                fprintf(stderr, "\n");
            }
        }
        if (hasPositional) fprintf(stderr, "\n");

        // Optional arguments section.
        fprintf(stderr, "options:\n");
        fprintf(stderr, "  %-32s %s\n", "-h, --help", "show this help message and exit");

        for (const auto& def : m_defs) {
            if (def.isPositional) continue;

            // Format: "-s, --stop_frame STOP_FRAME"
            std::string names;
            if (!def.shortName.empty()) names += "-" + def.shortName + ", ";
            else                        names += "    ";
            names += "--" + def.name;
            if (!def.isFlag) {
                std::string metavar = def.name;
                std::transform(metavar.begin(), metavar.end(), metavar.begin(), ::toupper);
                names += " " + metavar;
            }

            // Print with proper alignment.
            if (names.size() <= 32) {
                fprintf(stderr, "  %-32s", names.c_str());
            } else {
                fprintf(stderr, "  %s\n  %-32s", names.c_str(), "");
            }

            // Help text + defaults.
            std::string helpLine = def.help;
            if (def.required) helpLine += " (required)";
            if (!def.isFlag && !def.defaultValue.empty() && def.defaultValue != "false")
                helpLine += " (default: " + def.defaultValue + ")";
            fprintf(stderr, "%s\n", helpLine.c_str());
        }

        if (!m_error.empty()) {
            fprintf(stderr, "\nerror: %s\n", m_error.c_str());
        }
    }

    /// Alias for PrintHelp (backward compat).
    void PrintUsage() const { PrintHelp(); }

private:
    struct OptionDef {
        std::string name;         // canonical name (no dashes)
        std::string shortName;    // short char (no dash), empty = none
        std::string help;
        bool required = false;
        bool isFlag = false;
        bool isPositional = false;
        std::string defaultValue;
    };

    /// Normalize: strip leading dashes, replace '-' with '_'.
    static std::string Normalize(const std::string& s) {
        std::string out = StripDashes(s);
        std::replace(out.begin(), out.end(), '-', '_');
        return out;
    }

    /// Strip leading '-' or '--'.
    static std::string StripDashes(const std::string& s) {
        if (s.size() >= 2 && s[0] == '-' && s[1] == '-') return s.substr(2);
        if (s.size() >= 1 && s[0] == '-') return s.substr(1);
        return s;
    }

    /// Check if all chars in a string correspond to registered flags.
    bool AllShortFlags(const std::string& chars) const {
        for (char c : chars) {
            const OptionDef* def = FindDefByShort(std::string(1, c));
            if (!def || !def->isFlag) return false;
        }
        return true;
    }

    const OptionDef* FindDef(const std::string& normalizedName) const {
        for (const auto& d : m_defs) {
            if (!d.isPositional && Normalize(d.name) == normalizedName) return &d;
        }
        return nullptr;
    }

    const OptionDef* FindDefByShort(const std::string& shortStr) const {
        for (const auto& d : m_defs) {
            if (!d.shortName.empty() && d.shortName == shortStr) return &d;
        }
        return nullptr;
    }

    const OptionDef* GetPositionalAt(size_t idx) const {
        size_t count = 0;
        for (const auto& d : m_defs) {
            if (d.isPositional) {
                if (count == idx) return &d;
                count++;
            }
        }
        return nullptr;
    }

    std::string m_progName;
    std::vector<std::string> m_args;
    std::vector<OptionDef> m_defs;
    std::unordered_map<std::string, std::string> m_values;
    std::string m_error;
};

} // namespace app

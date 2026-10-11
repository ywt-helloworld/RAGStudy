#include "Text.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace rag {
std::vector<std::string> characters(const std::string &text) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < text.size();) {
        auto c = static_cast<unsigned char>(text[i]);
        std::size_t n =
            c < 128 ? 1
                    : (c >= 0xc2 && c <= 0xdf
                           ? 2
                           : (c >= 0xe0 && c <= 0xef ? 3 : (c >= 0xf0 && c <= 0xf4 ? 4 : 0)));
        if (!n || i + n > text.size())
            throw std::runtime_error("Invalid or incomplete UTF-8 input");
        for (std::size_t j = 1; j < n; ++j)
            if ((static_cast<unsigned char>(text[i + j]) & 0xc0) != 0x80)
                throw std::runtime_error("Invalid UTF-8 continuation");
        if (n >= 3) {
            auto d = static_cast<unsigned char>(text[i + 1]);
            if ((c == 0xe0 && d < 0xa0) || (c == 0xed && d >= 0xa0) || (c == 0xf0 && d < 0x90) ||
                (c == 0xf4 && d >= 0x90))
                throw std::runtime_error("Invalid UTF-8 code point");
        }
        out.push_back(text.substr(i, n));
        i += n;
    }
    return out;
}
std::vector<std::string> terms(const std::string &text) {
    auto chars = characters(text);
    std::vector<std::string> out;
    std::string word, prev;
    const std::string punctuation = "，。！？；：、（）【】《》“”‘’—…·　\n\r\t";
    auto flush = [&]() {
        if (!word.empty()) {
            out.push_back(word);
            word.clear();
        }
    };
    for (auto c : chars) {
        if (c.size() == 1) {
            unsigned char a = static_cast<unsigned char>(c[0]);
            if (std::isalnum(a) || a == '_') {
                word += static_cast<char>(std::tolower(a));
                prev.clear();
            } else {
                flush();
                prev.clear();
            }
        } else if (punctuation.find(c) != std::string::npos) {
            flush();
            prev.clear();
        } else {
            flush();
            if (!prev.empty())
                out.push_back(prev + c);
            out.push_back(c);
            prev = c;
        }
    }
    flush();
    return out;
}
std::string clip(const std::string &text, std::size_t count) {
    auto cs = characters(text);
    std::string out;
    for (std::size_t i = 0; i < std::min(count, cs.size()); ++i)
        out += cs[i];
    return out;
}
std::string readText(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot read: " + path.u8string());
    std::string s((std::istreambuf_iterator<char>(in)), {});
    if (s.compare(0, 3, "\xef\xbb\xbf") == 0)
        s.erase(0, 3);
    characters(s);
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    return s;
}
void writeJson(const std::filesystem::path &path, const nlohmann::json &data) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    auto temp = path;
    temp += ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("Cannot write: " + temp.u8string());
        out << data.dump(2);
        out.flush();
        if (!out)
            throw std::runtime_error("Write failed: " + temp.u8string());
    }
#ifdef _WIN32
    if (!MoveFileExW(temp.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Atomic file replacement failed: " + path.u8string());
#else
    std::filesystem::rename(temp, path);
#endif
}
std::string fingerprint(const std::string &text) {
    std::uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    std::ostringstream s;
    s << std::hex << h;
    return s.str();
}
std::string modelIdentity(const std::string &path) {
    auto p = std::filesystem::weakly_canonical(std::filesystem::u8path(path));
    if (!std::filesystem::is_regular_file(p))
        throw std::runtime_error("Model file missing: " + p.u8string());
    std::ifstream in(p, std::ios::binary);
    std::string head(65536, '\0');
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<std::size_t>(in.gcount()));
    return p.u8string() + ":" + std::to_string(std::filesystem::file_size(p)) + ":" +
           std::to_string(std::filesystem::last_write_time(p).time_since_epoch().count()) + ":" +
           fingerprint(head);
}
std::vector<Option> parseOptions(const std::string &q) {
    std::vector<std::pair<std::size_t, std::size_t>> marks;
    for (std::size_t i = 0; i + 1 < q.size(); ++i)
        if (q[i] >= 'A' && q[i] <= 'Z' &&
            (i == 0 || !std::isalnum(static_cast<unsigned char>(q[i - 1])))) {
            std::size_t j = i + 1;
            while (j < q.size() && (q[j] == ' ' || q[j] == '\t'))
                ++j;
            if (q.compare(j, 1, ".") == 0 || q.compare(j, 1, ")") == 0 ||
                q.compare(j, 3, "．") == 0 || q.compare(j, 3, "、") == 0) {
                std::size_t n = (static_cast<unsigned char>(q[j]) < 128) ? 1 : 3;
                marks.push_back({i, j + n});
            }
        }
    std::vector<Option> result;
    if (marks.size() < 2)
        return result;
    for (std::size_t k = 0; k < marks.size(); ++k) {
        if (q[marks[k].first] != static_cast<char>('A' + k))
            return {};
        auto end = k + 1 < marks.size() ? marks[k + 1].first : q.size();
        result.push_back(
            {q.substr(marks[k].first, 1), q.substr(marks[k].second, end - marks[k].second)});
    }
    return result;
}
std::string questionStem(const std::string &q) {
    for (const auto &marker : {"选择：", "选项：", "\nA.", "\nA、"}) {
        auto p = q.find(marker);
        if (p != std::string::npos)
            return q.substr(0, p);
    }
    return q;
}
} // namespace rag

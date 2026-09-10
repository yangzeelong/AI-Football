#ifndef MY_MESSAGE_HPP
#define MY_MESSAGE_HPP

#include <nexusflow/Message.hpp>

// SeqMessage 有值(m_dataVec)以及资源(m_mutex)
struct SeqMessage {
    void addData(const std::string& data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_dataVec.push_back(data);
    }

    std::string toString() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::ostringstream oss;
        oss << "SeqMessage: {size=" << m_dataVec.size() << ", elements=[";
        for (size_t idx = 0; idx < m_dataVec.size(); ++idx) {
            oss << m_dataVec[idx];
            if (idx < m_dataVec.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]}";
        return oss.str();
    }

    ~SeqMessage() = default;

private:
    mutable std::mutex m_mutex;
    std::vector<std::string> m_dataVec;
};

#endif
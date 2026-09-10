#ifndef NEXUSFLOW_MESSAGE_HPP
#define NEXUSFLOW_MESSAGE_HPP

#include <atomic>
#include <chrono> // <-- [新增] 包含 <chrono> 头文件
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept> // for std::runtime_error
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace nexusflow {

struct MessageMeta {
    uint64_t messageId; // The unique identifier for the message
    uint64_t timestamp; // The timestamp when the message was created
    std::string sourceName; // The name of the source of the message
};

/**
 * @class Message
 * @brief A type-erased, thread-safe, shared message container with Copy-On-Write (COW) semantics.
 *
 * This class holds an object of any type. Copying a Message is a cheap operation,
 * as it only duplicates a shared_ptr. The underlying data is shared among all copies.
 *
 * When a mutable reference to the data is requested via `Mut<T>()`, the container
 * checks if the data is shared. If it is, a deep copy of the data is created
 * transparently, ensuring that modifications do not affect other Message instances
 * sharing the original data. This "Copy-On-Write" behavior makes it highly efficient
 * for broadcast scenarios in multi-threaded pipelines while maintaining data integrity.
 */
class Message {
public:
    Message() noexcept = default;

    /**
     * @brief Constructs a Message with the given data.
     * @tparam T The type of the data. Must be copyable or movable, and not a reference or pointer.
     * @param data The data to store in the Message, moved or copied.
     * @param sourceName The name of the source of the message.
     */
    template <typename T, typename DT = typename std::decay<T>::type,
              typename = std::enable_if_t<!std::is_same<DT, Message>::value &&
                                          (std::is_copy_constructible<DT>::value || std::is_move_constructible<DT>::value) &&
                                          (!std::is_reference<DT>::value && !std::is_pointer<DT>::value)>>
    explicit Message(T&& data, std::string sourceName = "") {
        static_assert(std::is_copy_constructible<DT>::value, "Type must be copy-constructible for COW semantics.");
        static_assert(!std::is_reference<DT>::value && !std::is_pointer<DT>::value, "Type must not be a reference or pointer");

        // Create a shared_ptr to a Model<DT> containing the data
        m_content = std::make_shared<Model<DT>>(std::forward<T>(data));

        m_metaData.messageId = GenerateMessageId();
        m_metaData.timestamp = GetCurrentTimestamp();
        m_metaData.sourceName = std::move(sourceName);
    }

    // --- Copy, Move, and Default Operations ---
    // Default operations are perfect for COW: copying is a cheap shared_ptr copy.
    Message(const Message& other) = default;
    Message& operator=(const Message& other) = default;
    Message(Message&& other) noexcept = default;
    Message& operator=(Message&& other) noexcept = default;
    ~Message() = default;

    /**
     * @brief Creates an explicit deep copy of the message.
     * @return A new Message instance with its own unique copy of the data and metadata.
     */
    Message Clone() const {
        if (!HasData()) return {};

        Message clone;
        // Perform a deep copy of the content
        clone.m_content = m_content->Clone();
        // Copy metadata
        clone.m_metaData = m_metaData;
        return clone;
    }

    // --- Accessors ---
    inline bool HasData() const { return m_content != nullptr; }

    template <typename T>
    inline bool HasType() const {
        return HasData() && std::type_index(typeid(T)) == m_content->getTypeIndex();
    }

    inline const MessageMeta& GetMetaData() const { return m_metaData; }

    inline MessageMeta& MetaData() { return m_metaData; }

    // --- Rust-style COW Accessors: Reference and Pointer Based ---

    // --- Reference-based (throwing) accessors ---

    template <typename T>
    const T& Borrow() const {
        if (!HasType<T>()) {
            throw std::runtime_error("Message type mismatch or empty. Requested: " + std::string(typeid(T).name()) +
                                     ", Actual: " + (m_content ? m_content->getTypeIndex().name() : "[null]"));
        }
        return static_cast<const Model<T>*>(m_content.get())->m_data;
    }

    template <typename T>
    T& Mut() {
        if (!HasType<T>()) {
            throw std::runtime_error("Message type mismatch or empty. Requested: " + std::string(typeid(T).name()) +
                                     ", Actual: " + (m_content ? m_content->getTypeIndex().name() : "[null]"));
        }
        detach_if_shared();
        return static_cast<Model<T>*>(m_content.get())->m_data;
    }

    // --- Pointer-based (non-throwing) accessors ---

    /**
     * @brief Borrows an immutable pointer to the contained data.
     * This is a read-only operation and will NOT trigger a copy.
     * @tparam T The expected type of the data.
     * @return A const pointer to the data, or `nullptr` if the message is empty or the type does not match.
     */
    template <typename T>
    const T* BorrowPtr() const {
        if (!HasType<T>()) {
            return nullptr;
        }
        return &static_cast<const Model<T>*>(m_content.get())->m_data;
    }

    /**
     * @brief Gets a mutable pointer to the contained data.
     * This is a write operation. If the data is shared, it will trigger a deep copy (COW).
     * @tparam T The expected type of the data.
     * @return A mutable pointer to the data, or `nullptr` if the message is empty or the type does not match.
     */
    template <typename T>
    T* MutPtr() {
        if (!HasType<T>()) {
            return nullptr;
        }
        // The core COW logic: detach (clone) the data if it's shared.
        detach_if_shared();
        return &static_cast<Model<T>*>(m_content.get())->m_data;
    }

    // ToString for debugging/logging
    std::string ToString() const {
        std::ostringstream oss;
        oss << "Message ID: " << m_metaData.messageId << ", Timestamp: " << m_metaData.timestamp
            << ", Source: " << m_metaData.sourceName;
        if (m_content) {
            oss << ", Type: " << m_content->getTypeIndex().name() << ", SharedCount: " << m_content.use_count();
        } else {
            oss << ", Type: [null]";
        }
        return oss.str();
    }

private:
    // --- Internal Type-Erasure Implementation ---
    struct Concept {
        virtual ~Concept() = default;
        virtual std::type_index getTypeIndex() const noexcept = 0;
        virtual std::shared_ptr<Concept> Clone() const = 0; // For deep copying
    };

    template <typename T>
    struct Model final : Concept {
        // T must be copy-constructible to support the Clone operation for COW.
        static_assert(std::is_copy_constructible<T>::value, "Type T must be copy-constructible for Message's COW feature.");

        explicit Model(T data) : m_data(std::move(data)) {}

        std::type_index getTypeIndex() const noexcept override { return std::type_index(typeid(T)); }

        // Clone creates a new shared_ptr holding a new Model with a copy of m_data.
        std::shared_ptr<Concept> Clone() const override {
            return std::make_shared<Model<T>>(m_data); // Deep copy of m_data
        }

        T m_data; // The actual data is stored here.
    };

    // --- COW Helper ---
    /**
     * @brief If the content is shared (use_count > 1), replaces it with a deep copy.
     * This is the "Copy-On-Write" part of the implementation.
     */
    void detach_if_shared() {
        if (m_content && m_content.use_count() > 1) {
            m_content = m_content->Clone();
        }
    }

    // --- Static Helpers ---
    static uint64_t GenerateMessageId() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }

    static uint64_t GetCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }

private:
    // The single shared_ptr that manages the lifetime and sharing of the internal Model object.
    std::shared_ptr<Concept> m_content;
    MessageMeta m_metaData;
};

// Factory function for convenient construction
template <typename T>
static Message MakeMessage(T&& value, std::string source = "") {
    return Message(std::forward<T>(value), std::move(source));
}

} // namespace nexusflow

#endif // NEXUSFLOW_MESSAGE_HPP
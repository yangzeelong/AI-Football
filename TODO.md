
# NexusFlow 项目开发路线图与待办事项

本文档记录了 NexusFlow 框架未来的功能增强、API优化和架构改进计划。

---

### 核心功能增强 (Core Features)

-   [ ] **实现优雅停机 (Graceful Shutdown / Draining)**
    -   **目标**: 在调用 `Pipeline::Stop()` 时，确保上游模块先停止生产数据，并让管道中正在流动的数据被下游模块完全处理完毕，避免数据丢失。
    -   **任务**:
        -   [ ] **拓扑顺序**: 确保 `Pipeline::Impl` 能够随时从 `Graph` 对象获取模块的拓扑排序列表。
        -   [ ] **停止逻辑**: 修改 `Pipeline::Stop()` 的实现，使其按照**拓扑顺序的逆序**来停止 `Worker`。先停止上游生产者，并等待其输入队列为空后，再逐级停止下游消费者。
        -   [ ] **Worker 协作**: `Worker::Stop()` 方法需要更精细的实现，以支持“排空”模式。

---

### 监控与可观测性 (Monitoring & Observability)

-   [ ] **抽象监控与状态报告机制**
    -   **目标**: 提供一套机制来监控每个模块和队列的实时状态，便于调试和性能分析。
    -   **任务**:
        -   [ ] **定义指标 (Metrics)**: 确定需要监控的关键指标，例如：
            -   **模块层面**: 已处理消息数、处理错误数、平均处理延迟。
            -   **队列层面**: 当前队列长度、累计入队数、累计出队数（用于检测丢数据）。
        -   [ ] **设计 `MetricsCollector`**: 创建一个中心的监控收集器（可以是单例或由 `Pipeline` 持有），提供如 `incrementCounter(name)`, `setGauge(name, value)` 等接口。
        -   [ ] **集成到框架**:
            -   在 `Worker` 中，每次调用 `processBatch` 前后记录信息，并上报给 `MetricsCollector`。
            -   在 `Dispatcher` 中，每次 `send/broadcast` 时更新计数器。
            -   让 `ConcurrentQueue` 内部也持有原子计数器。
        -   [ ] **暴露接口 (Optional)**: 提供一种方式（如一个HTTP端点或日志定期打印）来查询和展示这些监控数据。

---

### API 优化与导出 (API Refinements)

-   [ ] **导出日志模块 API**
    -   **目标**: 允许用户在他们自己的模块实现中，使用框架统一的日志系统。
    -   **任务**:
        -   [ ] **创建公共头文件**: 创建一个新的头文件，例如 `include/nexusflow/logging.hpp`。
        -   [ ] **提供简洁的宏**: 在该头文件中，提供一组易于使用的日志宏，如 `NEXUSFLOW_LOG_INFO(...)`, `NEXUSFLOW_LOG_WARN(...)`。这些宏内部会调用您的日志系统实现（如 spdlog）。
        -   [ ] **隐藏实现**: 确保 `spdlog` 等第三方库的头文件不会被包含在公共的 `logging.hpp` 中，避免依赖泄露。
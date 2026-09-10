## 性能评测 (Performance Benchmark)

为了评估 NexusFlow 框架的性能开销，项目内提供了一个基准测试脚本。该脚本用于测量在特定拓扑结构下，框架的消息**吞吐量**和端到端**平均延迟**。

### 测试拓扑

评测采用了一个经典的 **扇出/扇入 (Fan-out/Fan-in)** 拓扑结构，以测试框架的并行分发和数据汇总能力：



### 结果

### 评测环境 (Benchmark Environment)

以下性能数据是在特定的硬件和软件环境下测得。在不同配置的机器上，您的结果可能会有所差异。

*   **CPU**: `Intel(R) Core(TM) i7-14700KF`
*   **核心数 (Cores)**: 28 (14 Cores x 2 Threads)
*   **L3 缓存 (L3 Cache)**: 33 MB
*   **内存 (Memory)**: 16 GB
*   **操作系统 (OS)**: Ubuntu 20.04.5 LTS WSL2
*   **内核版本 (Kernel)**: `6.6.87.2-microsoft-standard-WSL2`
*   **编译器 (Compiler)**: `g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0`

```bash 
# Release
--- Benchmark Results ---
Total Duration:   10.0005 s
Messages Processed: 1614764
-------------------------
Throughput:       161468 msg/s
Avg. Latency:     100.415 us
-------------------------

# Latest Release Result (2025.07.19)
--- Benchmark Results ---
Total Duration:   10.0004 s
Messages Processed: 2804190
-------------------------
Throughput:       280406 msg/s
Avg. Latency:     35.4208 us
-------------------------

```

### 说明

- **Total Duration**: 总测试时间。
- **Messages Processed**: 总处理消息数。
- **Throughput**: 吞吐量，即每秒处理的消息数。
- **Avg. Latency**: 平均延迟，即处理每条消息的平均时间。

### 注意

基准测试脚本仅用于评估 NexusFlow 框架的性能，实际使用时需要根据具体应用场景和需求进行调整。

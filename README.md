# NexusFlow: A High-Performance, Modern C++ Dataflow Pipeline Framework

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your_username/nexusflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++_Version](https://img.shields.io/badge/C++-14%2B-blue.svg)]()

**NexusFlow** is a high-performance, highly-decoupled, and dynamically configurable dataflow pipeline framework designed for modern C++. It aims to help developers easily build complex, multi-stage data processing tasks, such as video analytics pipelines, real-time ETL, industrial sensor data processing, and more.

The core idea of the framework is to decompose complex processing workflows into a series of independent **Modules**, which can then be connected like LEGO bricks into a powerful **Directed Acyclic Graph (DAG)** using simple configurations.

## Core Features

*   **Modular Design, Highly Decoupled**: Each module focuses exclusively on its own business logic. Modules communicate asynchronously via message queues, remaining completely unaware of each other's existence.
*   **Dynamic Configuration, Powerful & Flexible**: Use simple YAML files to define the entire pipeline topology, module types, and parameters. Reconfigure and refactor complex business flows without recompiling your code.
*   **Automatic Concurrency, Simplified Development**: The framework automatically assigns dedicated threads to drive each module (or uses a defined strategy). You can focus on your business logic without manually managing complex thread synchronization and lifecycles.
*   **Clean API, Easy to Use**: Provides two primary ways to construct a pipeline: a **programmatic approach (`PipelineBuilder`)** for rapid development and testing, and a **declarative approach (from YAML)** for production environments.
*   **High-Performance & Modern C++**: Core components are designed for performance, leveraging modern C++ features. Its core data container, `nexusflow::Message`, uses **Copy-On-Write (COW)** semantics for extreme efficiency in broadcast scenarios.

---

## Quick Start

The following example demonstrates how to build a pipeline with four modules: one input node distributes data to two parallel processing nodes, and one output node gathers the results.

### Option 1: Declarative Build via YAML (Recommended)

This is the most powerful and recommended way to use NexusFlow. It allows you to define your entire system through a configuration file, providing maximum flexibility.

#### 1. Create a Configuration File (`graph.yaml`)

```yaml
graph:
  name: "VideoAnalyticsPipeline"

  # 1. Define all module instances
  modules:
    - name: "InputNode"          # A unique instance name for the module
      class: "MockInputModule"   # The class name registered in the ModuleFactory

    - name: "ProcessNode1"
      class: "MockProcessModule"

    - name: "ProcessNode2"
      class: "MockProcessModule"

    - name: "OutputNode"
      class: "MockOutputModule"

  # 2. Define the connections (the dataflow topology)
  connections:
    - from: "InputNode"
      to: "ProcessNode1"
    - from: "InputNode"
      to: "ProcessNode2"
    - from: "ProcessNode1"
      to: "OutputNode"
    - from: "ProcessNode2"
      to: "OutputNode"
```

#### 2. Write the C++ Entry Point (`example.cpp`)

```cpp
#include "nexusflow/Pipeline.hpp"
#include "nexusflow/ModuleFactory.hpp"
#include "my_module/MockInputModule.hpp"    // Include your custom module headers
#include "my_module/MockProcessModule.hpp"
#include "my_module/MockOutputModule.hpp"
#include "utils/logging.hpp"               // Your logging utility

#include <iostream>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <memory>

using namespace nexusflow;

// Register all custom modules with the factory at application startup.
void registerAllModules() {
    NEXUSFLOW_REGISTER_MODULE(MockInputModule);
    NEXUSFLOW_REGISTER_MODULE(MockProcessModule);
    NEXUSFLOW_REGISTER_MODULE(MockOutputModule);
}

// A helper function to encapsulate the pipeline execution and teardown logic.
void executePipeline(Pipeline& pipeline) {
    if (pipeline.Init() != ErrorCode::SUCCESS) {
        throw std::runtime_error("Pipeline initialization failed.");
    }

    pipeline.Start();
    LOG_INFO("Pipeline running for 10 seconds...");
    std::this_thread::sleep_for(std::chrono::seconds(10));
    pipeline.Stop();
    pipeline.DeInit();
}

int main(int argc, char* argv[]) {
    // ... Initialize logger ...

    if (argc < 2) {
        LOG_ERROR("Usage: {} <path_to_graph_yaml>", argv[0]);
        return 1;
    }

    try {
        registerAllModules();
        std::string configPath = argv[1];
        
        // The Pipeline class provides a static factory method.
        auto pipeline = Pipeline::CreateFromYaml(configPath);
        if (!pipeline) {
            throw std::runtime_error("Failed to create pipeline from YAML config.");
        }

        executePipeline(*pipeline);

    } catch (const std::exception& e) {
        LOG_CRITICAL("An exception occurred: {}", e.what());
        return 1;
    }

    LOG_INFO("Execution finished successfully.");
    return 0;
}
```

### Option 2: Programmatic Build via `PipelineBuilder`

This approach is suitable for simple applications, unit tests, or scenarios where the topology needs to be generated dynamically in code.

```cpp
#include "nexusflow/Pipeline.hpp"
#include "nexusflow/PipelineBuilder.hpp"
// ... other necessary headers ...

void runWithBuilder() {
    auto inputModule    = std::make_shared<MockInputModule>("InputNode");
    auto process1Module = std::make_shared<MockProcessModule>("ProcessNode1");
    auto process2Module = std::make_shared<MockProcessModule>("ProcessNode2");
    auto outputModule   = std::make_shared<MockOutputModule>("OutputNode");

    auto pipeline = PipelineBuilder()
                        .AddModule(inputModule)
                        .AddModule(process1Module)
                        .AddModule(process2Module)
                        .AddModule(outputModule)
                        .Connect("InputNode", "ProcessNode1")
                        .Connect("InputNode", "ProcessNode2")
                        .Connect("ProcessNode1", "OutputNode")
                        .Connect("ProcessNode2", "OutputNode")
                        .Build();
    if (!pipeline) {
        throw std::runtime_error("Failed to build pipeline.");
    }
    
    executePipeline(*pipeline);
}
```

---

## Core Component: The `nexusflow::Message`

The `nexusflow::Message` is the universal, thread-safe data wrapper that flows through the pipeline. It is designed to be both highly performant and exceptionally safe.

### Design Philosophy

1.  **Type-Erasure**: A `Message` can hold an object of **any data type**, allowing different modules to communicate seamlessly.
2.  **Copy-On-Write (COW)**: This is the core performance feature.
    *   **Copying is cheap**: Copying a `Message` is a fast `shared_ptr` operation, ideal for broadcasting data to multiple downstream modules.
    *   **Mutation is safe**: When you attempt to *modify* a shared `Message`, the framework automatically performs a deep copy of the data *before* the modification. This ensures that changes in one branch of the pipeline do not accidentally affect others.
3.  **Expressive & Safe Accessors**: Instead of traditional getters, `Message` uses a `Borrow`/`Mut` naming convention inspired by Rust to make the developer's intent crystal clear.

### How to Use `Message`

#### Creating a Message
The recommended way is to use the `nexusflow::MakeMessage` factory function.
```cpp
#include "nexusflow/Message.hpp"

// Create a message containing a string
auto msg1 = nexusflow::MakeMessage(std::string("Hello, Pipeline!"));

// Create a message containing a vector, specifying its source
auto msg2 = nexusflow::MakeMessage(std::vector<int>{1, 2, 3}, "SensorModule");
```

#### Immutable Access (Borrowing)
Use "borrowing" for **read-only** access. This operation is always fast and **will never** trigger a copy.

*   **`Borrow<T>()`**: Returns a `const` reference. Throws `std::runtime_error` on type mismatch.
*   **`BorrowPtr<T>()`**: Returns a `const` pointer. Returns `nullptr` on type mismatch (no-throw).

```cpp
// Safely check for type with BorrowPtr
if (const auto* content_ptr = msg1.BorrowPtr<std::string>()) {
    std::cout << "Content: " << *content_ptr << std::endl;
} else {
    std::cout << "Message does not contain a string." << std::endl;
}
```

#### Mutable Access (Mutating)
Use "mutating" for **read-write** access. This **will trigger a Copy-On-Write** if the data is shared.

*   **`Mut<T>()`**: Returns a non-`const` reference. Throws `std::runtime_error` on type mismatch.
*   **`MutPtr<T>()`**: Returns a non-`const` pointer. Returns `nullptr` on type mismatch (no-throw).

```cpp
// Safely get a mutable pointer
if (auto* vec_ptr = msg2.MutPtr<std::vector<int>>()) {
    vec_ptr->push_back(4); // This might trigger a COW
}
```

### Copy-On-Write in Action

```cpp
// 1. Create an original message (SharedCount: 1)
auto original_msg = nexusflow::MakeMessage(std::vector<int>{10, 20, 30});

// 2. Create a shared copy (SharedCount becomes 2)
auto shared_copy = original_msg;

// 3. One module modifies its copy. This triggers COW.
// shared_copy creates its own data. Its SharedCount becomes 1.
// original_msg's SharedCount reverts to 1.
shared_copy.Mut<std::vector<int>>()[1] = 99;

// 4. The data has diverged safely.
std::cout << original_msg.Borrow<std::vector<int>>()[1]; // -> 20
std::cout << shared_copy.Borrow<std::vector<int>>()[1];  // -> 99
```

---

## How to Write a Custom Module

Creating a new module is a simple two-step process:

#### 1. Inherit from `nexusflow::Module` and implement `Process`

```cpp
// modules/MultiplierModule.hpp
#include "nexusflow/Module.hpp"
#include "nexusflow/Message.hpp"

class MultiplierModule : public nexusflow::Module {
public:
    // Every module must have a constructor that accepts a name.
    explicit MultiplierModule(std::string name) : nexusflow::Module(std::move(name)) {}

    // Implement the core processing logic.
    void Process(nexusflow::Message& msg) override {
        // Use MutPtr for safe, mutable access.
        if (auto* data = msg.MutPtr<int>()) {
            // Multiply the received number by 2.
            *data *= 2;
            
            // Broadcast the modified message downstream.
            // The message is implicitly shared, no extra copy needed.
            Broadcast(msg);
        }
    }
};
```

#### 2. Register it with the `ModuleFactory` in your application

```cpp
// main.cpp
#include "nexusflow/ModuleFactory.hpp"
#include "modules/MultiplierModule.hpp"

void registerAllModules() {
    // Register the MultiplierModule with the factory.
    NEXUSFLOW_REGISTER_MODULE(MultiplierModule);
}
```
That's it! You can now use `class: "MultiplierModule"` in your `graph.yaml` file.

## Building the Project

This project uses CMake for building.

```bash
# 1. Create a build directory
mkdir build
cd build

# 2. Run CMake to configure the project
# Assumes dependencies are managed by CMake's FetchContent or are system-installed.
cmake ..

# 3. Compile the project
make -j$(nproc)

# 4. Run the example
./examples/nexusflow_example path/to/your/graph.yaml
```

## Contributing

Contributions of any kind are welcome! If you have ideas, suggestions, or have found a bug, please feel free to open a Pull Request or create an Issue.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
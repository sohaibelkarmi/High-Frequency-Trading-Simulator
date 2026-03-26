cmake_minimum_required(VERSION 3.15)
project(hft_sim LANGUAGES CXX)

option(HFT_ENABLE_TESTS "Build unit tests" OFF)
option(ENABLE_DOCS "Generate documentation with Doxygen" OFF)
option(HFT_ENABLE_PROFILING "Enable in-simulator profiling instrumentation" OFF)
option(HFT_USE_LEGACY_LATENCY_QUEUE "Use legacy priority_queue latency queue" OFF)
if(HFT_ENABLE_TESTS)
    enable_testing()
endif()

find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HFT_WARNING_FLAGS -Wall -Wextra -Wpedantic)
set(HFT_OPTIMIZATION_FLAGS -O3 -march=native -fno-exceptions -fno-rtti -DNDEBUG)

function(hft_apply_compile_flags target)
    target_compile_options(${target} PRIVATE ${HFT_WARNING_FLAGS} ${HFT_OPTIMIZATION_FLAGS})
endfunction()

add_library(hft_core STATIC
    src/OrderBook.cpp
    src/adaptive_engine.cpp
    src/hawkes_core.cpp
    src/hawkes_engine.cpp
    src/engine_runtime.cpp
    src/simulator_core.cpp
    src/perf/Profiler.cpp
)
target_include_directories(hft_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(hft_core)
if(HFT_ENABLE_PROFILING)
    target_compile_definitions(hft_core PUBLIC HFT_ENABLE_PROFILING=1)
endif()
if(HFT_USE_LEGACY_LATENCY_QUEUE)
    target_compile_definitions(hft_core PUBLIC HFT_LEGACY_LATENCY_QUEUE=1)
endif()

add_library(order_book_bridge SHARED
    src/order_book_c_api.cpp
)
target_link_libraries(order_book_bridge PRIVATE hft_core)
target_include_directories(order_book_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(order_book_bridge)
set_target_properties(order_book_bridge PROPERTIES
    OUTPUT_NAME order_book_bridge
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_library(hawkes_bridge SHARED
    src/hawkes_c_api.cpp
)
target_link_libraries(hawkes_bridge PRIVATE hft_core)
target_include_directories(hawkes_bridge PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(hawkes_bridge)
set_target_properties(hawkes_bridge PROPERTIES
    OUTPUT_NAME hawkes_bridge
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_executable(hft_sim src/main.cpp)
target_link_libraries(hft_sim PRIVATE hft_core)
target_include_directories(hft_sim PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(hft_sim)

add_executable(hawkes_example src/hawkes_example.cpp)
target_link_libraries(hawkes_example PRIVATE hft_core)
target_include_directories(hawkes_example PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(hawkes_example)

add_executable(hawkes_validation src/hawkes_validation.cpp)
target_link_libraries(hawkes_validation PRIVATE hft_core)
target_include_directories(hawkes_validation PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(hawkes_validation)

# add_executable(execution_cost_runner src/execution_cost_runner.cpp)
# target_link_libraries(execution_cost_runner PRIVATE hft_core)
# target_include_directories(execution_cost_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
# hft_apply_compile_flags(execution_cost_runner)

add_executable(impact_calibration src/impact_calibration.cpp)
target_link_libraries(impact_calibration PRIVATE hft_core ZLIB::ZLIB)
target_include_directories(impact_calibration PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(impact_calibration)

add_executable(pnl_analysis src/pnl_analysis.cpp)
hft_apply_compile_flags(pnl_analysis)

add_executable(week8_integration src/week8_system_integration.cpp)
target_include_directories(week8_integration PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(week8_integration PRIVATE Threads::Threads)
hft_apply_compile_flags(week8_integration)

add_executable(hft_snap_host src/snap_engine_host.cpp)
target_link_libraries(hft_snap_host PRIVATE hft_core)
target_include_directories(hft_snap_host PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include/snap)
hft_apply_compile_flags(hft_snap_host)

add_executable(perf_benchmark src/perf/Benchmark.cpp)
target_link_libraries(perf_benchmark PRIVATE hft_core)
target_include_directories(perf_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(perf_benchmark)

# add_executable(parallel_benchmark src/perf/ParallelBenchmark.cpp)
# target_link_libraries(parallel_benchmark PRIVATE hft_core)
# target_include_directories(parallel_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
# hft_apply_compile_flags(parallel_benchmark)

add_executable(stress_test src/perf/StressTest.cpp)
target_link_libraries(stress_test PRIVATE hft_core)
target_include_directories(stress_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
hft_apply_compile_flags(stress_test)

if(ENABLE_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(
        docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

if(HFT_ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(order_tests tests/order_tests.cpp)
    target_link_libraries(order_tests PRIVATE Catch2::Catch2WithMain hft_core)
    target_include_directories(order_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME order_tests COMMAND order_tests)

    add_executable(execution_cost_test tests/execution_cost_test.cpp)
    target_link_libraries(execution_cost_test PRIVATE Catch2::Catch2WithMain hft_core)
    target_include_directories(execution_cost_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME execution_cost_test COMMAND execution_cost_test)
endif()

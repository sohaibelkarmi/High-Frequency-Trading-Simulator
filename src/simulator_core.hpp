#pragma once

#include "hawkes_engine.hpp"
#include "latency_model.hpp"
#include "OrderBook.hpp"
#include "execution_cost.hpp"
#include "memory_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace simulator {

struct SimulationConfig {
    std::vector<double> mu;
    ExponentialHawkesProcess::Matrix alpha;
    ExponentialHawkesProcess::Matrix beta;
    double session_length{60.0};
    std::size_t max_events{0};
    double latency_mean_us{120.0};
    std::uint64_t seed{1337};
    std::filesystem::path event_log_path{};
    ExecutionCostConfig execution_cost{};
    double base_aggressiveness{1.0};
    std::int32_t aggressive_order_size{1};
    bool use_snap_shm{false};
};

struct IntensitySample {
    double time{};
    SnapIntensityBuffer lambda;
};

struct ArrivalRecord {
    double raw_time{};
    double arrival_time{};
    double latency{};
    std::size_t dimension{};
    double intensity_total{};
    double intensity_dimension{};
};

struct SimulationResult {
    double horizon{};
    std::vector<IntensitySample> intensity_trace;
    std::vector<ArrivalRecord> arrivals;
    double mean_interarrival{};
    double variance_interarrival{};
    double mean_intensity{};
    std::vector<ExecutionRecord> executions;
    double cumulative_execution_cost{};
    double cumulative_temporary_cost{};
    double cumulative_permanent_cost{};
    double cumulative_shortfall{};
    double mean_slippage{};
    double cost_variance{};
    double mean_aggressiveness{};
};

class SimulatorCore {
public:
    explicit SimulatorCore(SimulationConfig config);

    SimulationResult run();

private:
    struct PendingPayload {
        HawkesEvent event;
    };

    void seed_order_book(OrderBook& book);
    void update_summary_metrics(SimulationResult& result) const;
    void write_event_log(const SimulationResult& result) const;

    SimulationConfig config_;
    ExponentialLatencyModel latency_model_;
    mutable std::uint64_t next_order_id_{1};
    ExecutionEngine execution_engine_;
};

SimulationConfig default_btcusdt_config();

} // namespace simulator

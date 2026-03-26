#pragma once

#include "OrderBook.hpp"
#include "event_engine.hpp"
#include "execution_cost.hpp"
#include "latency_model.hpp"
#include "simulator_core.hpp"
#include "thread_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include <boost/lockfree/spsc_queue.hpp>

namespace simulator {

struct MatchEnvelope {
    ArrivalRecord arrival;
    ExecutionRecord execution;
    IntensitySample intensity;
    std::chrono::high_resolution_clock::time_point matched_at{};
    std::chrono::high_resolution_clock::time_point produced_at{};
    double event_to_match_latency_us{0.0};
    bool is_termination{false};
};

using MatchQueue = boost::lockfree::spsc_queue<MatchEnvelope>;
// Snap SHM Match Queue will be added as a separate implementation or injected.

template <typename InQueue = EventQueue, typename OutQueue = MatchQueue>
class MatchingEngine {
public:
    using SeedOrderBookFn = std::function<void(OrderBook&)>;

    MatchingEngine(const SimulationConfig& config,
                   InQueue& input_queue,
                   OutQueue& output_queue,
                   ExecutionEngine& execution_engine,
                   SeedOrderBookFn seed_fn,
                   std::atomic<std::uint64_t>& next_order_id,
                   std::atomic<bool>& shutdown_signal,
                   std::atomic<bool>& event_stream_finished,
                   std::atomic<bool>& finished_flag,
                   concurrency::ThreadConfig thread_config);
    ~MatchingEngine();

    void start();
    void join();

    [[nodiscard]] concurrency::ThreadStats stats() const noexcept;
    [[nodiscard]] std::size_t processed_events() const noexcept;
    [[nodiscard]] std::size_t queue_retries() const noexcept;

private:
    void run();
    bool flush_ready(double time_limit);
    void push_output(MatchEnvelope&& envelope);
    Order create_market_order(std::size_t dimension,
                              double reference_price,
                              std::int32_t quantity,
                              std::int64_t timestamp_ns,
                              Side side);
    double determine_reference_price(const OrderBook& book, Side side) const;
    std::int64_t next_order_id();

    const SimulationConfig& config_;
    InQueue& input_queue_;
    OutQueue& output_queue_;
    ExecutionEngine& execution_engine_;
    SeedOrderBookFn seed_fn_;
    std::atomic<std::uint64_t>& next_order_id_;
    std::atomic<bool>& shutdown_signal_;
    std::atomic<bool>& event_stream_finished_;
    std::atomic<bool>& finished_flag_;
    concurrency::ThreadConfig thread_config_;
    std::thread thread_;
    concurrency::ThreadStats stats_{};
    std::atomic<std::size_t> processed_{0};
    std::atomic<std::size_t> push_retries_{0};

    struct PendingPayload {
        EventEnvelope envelope;
    };

    OrderBook book_;
    ExponentialLatencyModel latency_model_;
    std::mt19937_64 rng_;
    LatencyQueue<PendingPayload> latency_queue_;
    std::vector<LatencyQueue<PendingPayload>::Entry> ready_buffer_;
    std::vector<OrderBook::Fill> fills_buffer_;
};

template <typename InQueue, typename OutQueue>
MatchingEngine<InQueue, OutQueue>::MatchingEngine(const SimulationConfig& config,
                               InQueue& input_queue,
                               OutQueue& output_queue,
                               ExecutionEngine& execution_engine,
                               SeedOrderBookFn seed_fn,
                               std::atomic<std::uint64_t>& next_order_id,
                               std::atomic<bool>& shutdown_signal,
                               std::atomic<bool>& event_stream_finished,
                               std::atomic<bool>& finished_flag,
                               concurrency::ThreadConfig thread_config)
    : config_(config),
      input_queue_(input_queue),
      output_queue_(output_queue),
      execution_engine_(execution_engine),
      seed_fn_(std::move(seed_fn)),
      next_order_id_(next_order_id),
      shutdown_signal_(shutdown_signal),
      event_stream_finished_(event_stream_finished),
      finished_flag_(finished_flag),
      thread_config_(std::move(thread_config)),
      latency_model_(config.latency_mean_us),
      rng_(config.seed ^ 0xA4B1C3D5E7F91324ULL) {
    if (thread_config_.name.empty()) {
        thread_config_.name = "matching-engine";
    }
}

template <typename InQueue, typename OutQueue>
MatchingEngine<InQueue, OutQueue>::~MatchingEngine() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

template <typename InQueue, typename OutQueue>
void MatchingEngine<InQueue, OutQueue>::start() {
    thread_ = std::thread(&MatchingEngine::run, this);
}

template <typename InQueue, typename OutQueue>
void MatchingEngine<InQueue, OutQueue>::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

template <typename InQueue, typename OutQueue>
void MatchingEngine<InQueue, OutQueue>::run() {
    concurrency::apply_thread_config(thread_config_);
    const auto timing = concurrency::begin_timing();
    seed_fn_(book_);
    bool limit_reached = false;
    double last_event_time = 0.0;
    while (!shutdown_signal_.load(std::memory_order_relaxed) && !limit_reached) {
        EventEnvelope envelope;
        if (input_queue_.pop(envelope)) {
            if (envelope.is_termination) break;
            last_event_time = envelope.event.time;
            PendingPayload payload;
            payload.envelope = std::move(envelope);
            const double delay = latency_model_.sample_delay(rng_);
            const double ready_time = payload.envelope.event.time + delay;
            latency_queue_.push(ready_time, delay, std::move(payload));
            limit_reached = flush_ready(last_event_time);
        } else {
            if (event_stream_finished_.load(std::memory_order_acquire)) break;
            if (!latency_queue_.empty()) limit_reached = flush_ready(last_event_time);
            else std::this_thread::yield();
        }
    }
    flush_ready(config_.session_length);
    MatchEnvelope sentinel{};
    sentinel.is_termination = true;
    push_output(std::move(sentinel));
    stats_ = concurrency::end_timing(timing);
    finished_flag_.store(true, std::memory_order_release);
}

template <typename InQueue, typename OutQueue>
void MatchingEngine<InQueue, OutQueue>::push_output(MatchEnvelope&& envelope) {
    while (!output_queue_.push(envelope)) {
        if (shutdown_signal_.load(std::memory_order_relaxed)) return;
        push_retries_.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    }
}

template <typename InQueue, typename OutQueue>
bool MatchingEngine<InQueue, OutQueue>::flush_ready(double time_limit) {
    ready_buffer_.clear();
    latency_queue_.drain_up_to(time_limit, std::back_inserter(ready_buffer_));
    for (auto& entry : ready_buffer_) {
        const auto matched_at = std::chrono::high_resolution_clock::now();
        const EventEnvelope& env = entry.payload.envelope;
        ArrivalRecord arrival{env.event.time, entry.ready_time, entry.delay, env.event.dimension, env.event.intensity_total, env.event.intensity_dimension};
        const Side side = (arrival.dimension == 0 ? Side::Buy : Side::Sell);
        const std::int32_t quantity = std::max<std::int32_t>(1, config_.aggressive_order_size);
        book_.match_into(side, 0.0, quantity, fills_buffer_);
        ExecutionRecord exec_record = execution_engine_.record_execution(Order{0, side, 0.0, quantity, 0}, 0.0, config_.base_aggressiveness, fills_buffer_);
        MatchEnvelope outbound{arrival, exec_record, {env.event.time, env.event.intensities}, matched_at, env.produced_at, std::chrono::duration<double, std::micro>(matched_at - env.produced_at).count(), false};
        push_output(std::move(outbound));
        if (config_.max_events > 0 && processed_.fetch_add(1, std::memory_order_relaxed) + 1 >= config_.max_events) return true;
    }
    return false;
}

} // namespace simulator

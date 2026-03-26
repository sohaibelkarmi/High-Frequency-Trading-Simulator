#pragma once

#include "hawkes_engine.hpp"
#include "thread_utils.hpp"
#include "memory_pool.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <boost/lockfree/spsc_queue.hpp>

namespace simulator {

struct SimulationConfig;

struct EventEnvelope {
    HawkesEvent event{};
    std::chrono::high_resolution_clock::time_point produced_at{};
    bool is_termination{false};
};

using EventQueue = boost::lockfree::spsc_queue<EventEnvelope>;

template <typename Q = EventQueue>
class EventEngine {
public:
    EventEngine(const SimulationConfig& config,
                Q& queue,
                std::atomic<bool>& shutdown_signal,
                std::atomic<bool>& finished_flag,
                concurrency::ThreadConfig thread_config);
    ~EventEngine();

    void start();
    void join();

    [[nodiscard]] concurrency::ThreadStats stats() const noexcept;
    [[nodiscard]] std::size_t produced_events() const noexcept;
    [[nodiscard]] std::size_t queue_retries() const noexcept;

private:
    void run();
    bool should_stop(double event_time, std::size_t produced) const;

    const SimulationConfig& config_;
    Q& queue_;
    std::atomic<bool>& shutdown_signal_;
    std::atomic<bool>& finished_flag_;
    concurrency::ThreadConfig thread_config_;
    std::thread thread_;
    concurrency::ThreadStats stats_{};
    std::atomic<std::size_t> produced_{0};
    std::atomic<std::size_t> push_retries_{0};
};

template <typename Q>
EventEngine<Q>::EventEngine(const SimulationConfig& config,
                         Q& queue,
                         std::atomic<bool>& shutdown_signal,
                         std::atomic<bool>& finished_flag,
                         concurrency::ThreadConfig thread_config)
    : config_(config), queue_(queue), shutdown_signal_(shutdown_signal),
      finished_flag_(finished_flag), thread_config_(std::move(thread_config)) {}

template <typename Q>
EventEngine<Q>::~EventEngine() { if (thread_.joinable()) thread_.join(); }

template <typename Q>
void EventEngine<Q>::start() { thread_ = std::thread(&EventEngine::run, this); }

template <typename Q>
void EventEngine<Q>::join() { if (thread_.joinable()) thread_.join(); }

template <typename Q>
void EventEngine<Q>::run() {
    concurrency::apply_thread_config(thread_config_);
    const auto timing = concurrency::begin_timing();
    std::mt19937_64 rng(config_.seed);
    ExponentialHawkesProcess process(config_.mu, config_.alpha, config_.beta);
    process.reset(0.0);
    while (!shutdown_signal_.load(std::memory_order_relaxed)) {
        HawkesEvent next = process.sample_next(rng);
        EventEnvelope envelope{next, std::chrono::high_resolution_clock::now(), false};
        const auto& lambda_snapshot = process.intensities();
        envelope.event.intensities.size = std::min(lambda_snapshot.size(), MAX_HAWKES_DIM);
        for (std::size_t i = 0; i < envelope.event.intensities.size; ++i) envelope.event.intensities.data[i] = lambda_snapshot[i];
        while (!queue_.push(envelope)) {
            if (shutdown_signal_.load(std::memory_order_relaxed)) break;
            push_retries_.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
        if (next.time >= config_.session_length || (config_.max_events > 0 && produced_.fetch_add(1, std::memory_order_relaxed) + 1 >= config_.max_events)) break;
    }
    EventEnvelope sentinel{HawkesEvent{}, std::chrono::high_resolution_clock::now(), true};
    while (!queue_.push(sentinel)) {
        if (shutdown_signal_.load(std::memory_order_relaxed)) break;
        push_retries_.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    }
    stats_ = concurrency::end_timing(timing);
    finished_flag_.store(true, std::memory_order_release);
}

} // namespace simulator

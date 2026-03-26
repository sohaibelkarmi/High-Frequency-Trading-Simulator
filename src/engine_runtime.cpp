#include "engine_runtime.hpp"
#include "snap/snap_shm_queue.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sys/resource.h>
#include <thread>
#include <utility>

namespace simulator {

namespace {

struct UsageSnapshot {
    double cpu_seconds{0.0};
    double rss_mb{0.0};
};

UsageSnapshot sample_usage() {
    ::rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return UsageSnapshot{};
    }
    const auto to_seconds = [](const timeval& tv) {
        return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) * 1e-6;
    };
    UsageSnapshot snapshot;
    snapshot.cpu_seconds = to_seconds(usage.ru_utime) + to_seconds(usage.ru_stime);
#ifdef __APPLE__
    snapshot.rss_mb = static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    snapshot.rss_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
    return snapshot;
}

class ResourceMonitor {
public:
    explicit ResourceMonitor(std::chrono::milliseconds interval = std::chrono::milliseconds(500))
        : interval_(interval) {}

    void start() {
        samples_.clear();
        running_.store(true, std::memory_order_release);
        start_time_ = std::chrono::steady_clock::now();
        prev_time_ = start_time_;
        prev_usage_ = sample_usage();
        thread_ = std::thread(&ResourceMonitor::run, this);
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    const std::vector<EngineRuntimeMetrics::ResourceSample>& samples() const noexcept {
        return samples_;
    }

private:
    void run() {
        while (running_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(interval_);
            const auto now = std::chrono::steady_clock::now();
            UsageSnapshot current = sample_usage();

            const double elapsed = std::chrono::duration<double>(now - start_time_).count();
            const double wall_delta = std::chrono::duration<double>(now - prev_time_).count();
            const double cpu_delta = current.cpu_seconds - prev_usage_.cpu_seconds;

            double cpu_util = 0.0;
            if (wall_delta > 1e-9 && cpu_delta >= 0.0) {
                cpu_util = cpu_delta / wall_delta;
            }

            samples_.push_back({elapsed, current.cpu_seconds, cpu_util, current.rss_mb});

            prev_time_ = now;
            prev_usage_ = current;
        }
    }

    std::chrono::milliseconds interval_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::chrono::steady_clock::time_point start_time_{};
    std::chrono::steady_clock::time_point prev_time_{};
    UsageSnapshot prev_usage_{};
    std::vector<EngineRuntimeMetrics::ResourceSample> samples_;
};

constexpr double kMidPrice = 27'000.0;
constexpr double kTickSize = 0.5;
constexpr std::int32_t kBaseQuantity = 5;
constexpr std::size_t kSeedDepth = 8;

} // namespace

EngineRuntime::EngineRuntime(SimulationConfig config,
                             std::size_t queue_capacity,
                             concurrency::ThreadConfig event_thread,
                             concurrency::ThreadConfig match_thread,
                             concurrency::ThreadConfig log_thread)
    : config_(std::move(config)),
      queue_capacity_(queue_capacity),
      event_thread_config_(std::move(event_thread)),
      match_thread_config_(std::move(match_thread)),
      log_thread_config_(std::move(log_thread)) {
    if (event_thread_config_.name.empty()) {
        event_thread_config_.name = "runtime-event";
    }
    if (match_thread_config_.name.empty()) {
        match_thread_config_.name = "runtime-match";
    }
    if (log_thread_config_.name.empty()) {
        log_thread_config_.name = "runtime-log";
    }
}

void EngineRuntime::request_shutdown() {
    shutdown_.store(true, std::memory_order_release);
}

void EngineRuntime::seed_order_book(OrderBook& book) {
    for (std::size_t level = 0; level < kSeedDepth; ++level) {
        const double price_offset = static_cast<double>(level) * kTickSize;
        const std::int32_t qty = kBaseQuantity + static_cast<std::int32_t>(level);
        book.addLimitOrder(Order{
            static_cast<std::int64_t>(next_order_id_.fetch_add(1, std::memory_order_relaxed)),
            Side::Buy,
            kMidPrice - price_offset,
            qty,
            0
        });
        book.addLimitOrder(Order{
            static_cast<std::int64_t>(next_order_id_.fetch_add(1, std::memory_order_relaxed)),
            Side::Sell,
            kMidPrice + price_offset,
            qty,
            0
        });
    }
}

EngineRuntimeMetrics EngineRuntime::run() {
    shutdown_.store(false, std::memory_order_release);
    next_order_id_.store(1, std::memory_order_relaxed);

    EventQueue event_queue(queue_capacity_);
    MatchQueue match_queue(queue_capacity_);

    std::atomic<bool> event_finished{false};
    std::atomic<bool> matcher_finished{false};

    ExecutionEngine execution_engine(config_.execution_cost, config_.max_events);
    if (config_.max_events > 0) {
        execution_engine.reserve(config_.max_events);
    }

    MatchingEngine::SeedOrderBookFn seed_fn = [this](OrderBook& book) {
        seed_order_book(book);
    };

    ResourceMonitor monitor;
    monitor.start();

    if (config_.use_snap_shm) {
        SnapShmQueue<EventEnvelope> event_queue("hft_event_queue", true, queue_capacity_);
        SnapShmQueue<MatchEnvelope> match_queue("hft_match_queue", true, queue_capacity_);

        EventEngine<SnapShmQueue<EventEnvelope>> event_engine(config_, event_queue, shutdown_, event_finished, event_thread_config_);
        MatchingEngine<SnapShmQueue<EventEnvelope>, SnapShmQueue<MatchEnvelope>> matching_engine(config_, event_queue, match_queue, execution_engine, std::move(seed_fn), next_order_id_, shutdown_, event_finished, matcher_finished, match_thread_config_);
        
        // Logger needs to be updated or we use a separate consumer process for matches.
        // For now, we'll let the engine run.
        event_engine.start();
        matching_engine.start();
        event_engine.join();
        event_finished.store(true, std::memory_order_release);
        matching_engine.join();
        
        metrics.events_generated = event_engine.produced_events();
        metrics.events_processed = matching_engine.processed_events();
    } else {
        EventQueue event_queue(queue_capacity_);
        MatchQueue match_queue(queue_capacity_);

        EventEngine<EventQueue> event_engine(config_, event_queue, shutdown_, event_finished, event_thread_config_);
        MatchingEngine<EventQueue, MatchQueue> matching_engine(config_, event_queue, match_queue, execution_engine, std::move(seed_fn), next_order_id_, shutdown_, event_finished, matcher_finished, match_thread_config_);
        
        // AnalyticsLogger logger(match_queue, shutdown_, matcher_finished, log_thread_config_);

        event_engine.start();
        matching_engine.start();
        // logger.start();

        event_engine.join();
        event_finished.store(true, std::memory_order_release);
        matching_engine.join();
        matcher_finished.store(true, std::memory_order_release);
        // logger.join();

        metrics.events_generated = event_engine.produced_events();
        metrics.events_processed = matching_engine.processed_events();
        metrics.event_queue_retries = event_engine.queue_retries();
        metrics.match_queue_retries = matching_engine.queue_retries();
        // metrics.logger_metrics = logger.metrics();
        metrics.event_thread = event_engine.stats();
        metrics.match_thread = matching_engine.stats();
        // metrics.logger_thread = logger.stats();
    }
    monitor.stop();

    metrics.event_queue_retries = event_engine.queue_retries();
    metrics.match_queue_retries = matching_engine.queue_retries();
    metrics.logger_metrics = logger.metrics();
    metrics.event_thread = event_engine.stats();
    metrics.match_thread = matching_engine.stats();
    metrics.logger_thread = logger.stats();
    metrics.resource_samples = monitor.samples();
    return metrics;
}

} // namespace simulator

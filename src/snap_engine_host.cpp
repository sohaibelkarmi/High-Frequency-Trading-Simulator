#include "engine_runtime.hpp"
#include "snap/snap_shm_queue.hpp"
#include <iostream>
#include <csignal>

using namespace simulator;

std::atomic<bool> g_shutdown{false};

void signal_handler(int) {
    g_shutdown.store(true);
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::cout << "[SnapHost] Starting HFT-Simulator Matcher Host via Snap SHM...\n";

    SimulationConfig config = default_btcusdt_config();
    config.use_snap_shm = true;

    // In a real multi-process setup, we would only start the MatchingEngine here,
    // and wait for events from the SnapShmQueue ("hft_event_queue").
    
    SnapShmQueue<EventEnvelope> input_queue("hft_event_queue", false); // Consumer
    SnapShmQueue<MatchEnvelope> output_queue("hft_match_queue", true); // Producer

    std::atomic<bool> event_stream_finished{false};
    std::atomic<bool> matcher_finished{false};
    std::atomic<uint64_t> next_order_id{1000000}; // Start high to avoid collision

    ExecutionEngine execution_engine(config.execution_cost);
    
    MatchingEngine<SnapShmQueue<EventEnvelope>, SnapShmQueue<MatchEnvelope>> matching_engine(
        config,
        input_queue,
        output_queue,
        execution_engine,
        [](OrderBook& book) { /* Optional: seed from SHM as well */ },
        next_order_id,
        g_shutdown,
        event_stream_finished,
        matcher_finished,
        {}
    );

    std::cout << "[SnapHost] Matcher is now listening on 'hft_event_queue'...\n";
    matching_engine.start();
    matching_engine.join();

    std::cout << "[SnapHost] Matcher shut down. Processed " << matching_engine.processed_events() << " events.\n";
    return 0;
}

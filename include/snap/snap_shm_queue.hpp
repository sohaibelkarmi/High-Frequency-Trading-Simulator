#pragma once

#include "snap/includes/shm_link.hpp"
#include <string>
#include <optional>

namespace simulator {

template <typename T>
class SnapShmQueue {
public:
    SnapShmQueue(const std::string& name, bool producer, std::size_t capacity = 1024)
        : name_(name), is_producer_(producer) {
        if (producer) {
            link_ = snap::ShmLink::create(name, capacity * sizeof(T));
        } else {
            // In a real scenario, the consumer would wait for the producer to create it.
            // For now, we assume the producer starts first or we retry.
            link_ = snap::ShmLink::open(name);
        }
    }

    bool push(const T& item) {
        if (!link_ || !is_producer_) return false;
        return link_->send(&item, sizeof(T));
    }

    bool pop(T& item) {
        if (!link_ || is_producer_) return false;
        std::size_t received = 0;
        return link_->recv(&item, sizeof(T), received) && received == sizeof(T);
    }

    bool empty() const {
        // Snap ShmLink doesn't have a direct 'empty' check without receiving.
        // For simplicity in this integration, we'll assume it's used in a context
        // where 'pop' is called until it fails.
        return false; 
    }

private:
    std::string name_;
    bool is_producer_;
    std::unique_ptr<snap::ShmLink> link_;
};

} // namespace simulator

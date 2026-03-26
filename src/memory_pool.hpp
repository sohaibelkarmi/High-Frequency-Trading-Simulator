#pragma once

#include <cstddef>
#include <memory_resource>
#include <vector>

namespace simulator {

template <typename T>
class PoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    PoolAllocator() noexcept
        : resource_(pool_resource()) {}

    template <typename U>
    PoolAllocator(const PoolAllocator<U>& other) noexcept
        : resource_(other.resource_) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        return static_cast<T*>(resource_->allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        resource_->deallocate(p, n * sizeof(T), alignof(T));
    }

    template <typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };

    bool operator==(const PoolAllocator& other) const noexcept {
        return resource_ == other.resource_;
    }

    bool operator!=(const PoolAllocator& other) const noexcept {
        return !(*this == other);
    }

private:
    template <typename>
    friend class PoolAllocator;

    static std::pmr::memory_resource* pool_resource() {
        static std::pmr::synchronized_pool_resource resource;
        return &resource;
    }

    std::pmr::memory_resource* resource_;
};

template <typename T>
using PooledVector = std::vector<T, PoolAllocator<T>>;

using IntensityBuffer = PooledVector<double>;

static constexpr std::size_t MAX_HAWKES_DIM = 16;
struct SnapIntensityBuffer {
    double data[MAX_HAWKES_DIM];
    std::size_t size;
};

} // namespace simulator

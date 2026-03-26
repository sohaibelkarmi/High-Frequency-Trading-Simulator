#pragma once

#include "Order.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

class OrderBook {
public:
    using OrderId   = std::int64_t;
    using PriceTick = std::int64_t;  // scaled price (e.g. price * 10000)

    struct LevelSnapshot {
        double price;                     // display price
        std::int64_t totalQuantity;       // aggregate resting size
        std::vector<Order> resting;       // FIFO orders at this price
        Side side;                        // side this level belongs to
    };

    struct Fill {
        Order order;                      // resting order metadata after the match
        std::int32_t executedQuantity;    // quantity executed in this fill
        double fillPrice;                 // execution price applied
    };

    void addLimitOrder(const Order& order);
    bool cancel(OrderId id);
    std::vector<Fill> match(
        Side aggressor,
        double price,
        std::int32_t quantity
    );

    void match_into(
        Side aggressor,
        double price,
        std::int32_t quantity,
        std::vector<Fill>& fills
    );

    std::optional<Order> bestBid() const;
    std::optional<Order> bestAsk() const;

    std::vector<LevelSnapshot> levels(std::size_t depth) const;
    void printTopLevels(std::size_t depth) const;

private:
    struct OrderNode {
        Order data{};
        OrderNode* next{nullptr};
        OrderNode* prev{nullptr};
    };

    class OrderPool {
    public:
        explicit OrderPool(std::size_t blockSize = 1024);
        OrderNode* acquire(const Order& order);
        void release(OrderNode* node);

    private:
        void expand();

        std::size_t blockSize_;
        std::vector<std::unique_ptr<OrderNode[]>> blocks_;
        std::vector<OrderNode*> freeList_;
    };

    struct PriceLevel {
        OrderNode* head{nullptr};
        OrderNode* tail{nullptr};
        std::int64_t totalQuantity{0};

        inline void append(OrderNode* node) noexcept {
            node->next = nullptr;
            node->prev = tail;
            if (tail) {
                tail->next = node;
            } else {
                head = node;
            }
            totalQuantity += node->data.quantity;
            tail = node;
        }

        inline void remove(OrderNode* node) noexcept {
            if (node->prev) {
                node->prev->next = node->next;
            } else {
                head = node->next;
            }
            if (node->next) {
                node->next->prev = node->prev;
            } else {
                tail = node->prev;
            }
            totalQuantity -= node->data.quantity;
            if (totalQuantity < 0) totalQuantity = 0;
        }

        [[nodiscard]] inline bool empty() const noexcept { return head == nullptr; }
    };

    template <typename Compare>
    class FlatBook {
    public:
        explicit FlatBook(Compare comp = Compare{}) : comp_(comp) {}

        PriceLevel& levelFor(PriceTick price);
        PriceLevel* levelOf(PriceTick price);
        const PriceLevel* levelOf(PriceTick price) const;
        const PriceLevel* bestLevel() const;
        bool eraseIfEmpty(PriceTick price);
        bool empty() const noexcept { return entries_.empty(); }
        struct Entry {
            PriceTick price;
            std::unique_ptr<PriceLevel> level;
        };
        const std::vector<Entry>& entries() const noexcept { return entries_; }
        std::vector<Entry>& entries() noexcept { return entries_; }

    private:
        std::size_t findInsertPos(PriceTick price) const;
        void refreshIndices(std::size_t start);

        Compare comp_;
        std::vector<Entry> entries_;
        std::unordered_map<PriceTick, std::size_t> index_;
    };

    struct Locator {
        Side side;
        PriceTick price;
        OrderNode* node;
    };

    class LocatorTable {
    public:
        void set(OrderId id, const Locator& locator) {
            if (id < 0) {
                return;
            }
            const std::size_t idx = static_cast<std::size_t>(id);
            if (idx >= storage_.size()) {
                storage_.resize(idx + 1);
                active_.resize(idx + 1, 0);
            }
            storage_[idx] = locator;
            active_[idx] = 1;
        }

        Locator* find(OrderId id) {
            if (id < 0) {
                return nullptr;
            }
            const std::size_t idx = static_cast<std::size_t>(id);
            if (idx >= storage_.size() || active_[idx] == 0) {
                return nullptr;
            }
            return &storage_[idx];
        }

        const Locator* find(OrderId id) const {
            if (id < 0) {
                return nullptr;
            }
            const std::size_t idx = static_cast<std::size_t>(id);
            if (idx >= storage_.size() || active_[idx] == 0) {
                return nullptr;
            }
            return &storage_[idx];
        }

        bool erase(OrderId id) {
            if (id < 0) {
                return false;
            }
            const std::size_t idx = static_cast<std::size_t>(id);
            if (idx >= storage_.size() || active_[idx] == 0) {
                return false;
            }
            active_[idx] = 0;
            storage_[idx].node = nullptr;
            return true;
        }

    private:
        std::vector<Locator> storage_;
        std::vector<std::uint8_t> active_;
    };

    using BidBook = FlatBook<std::greater<PriceTick>>;
    using AskBook = FlatBook<std::less<PriceTick>>;

    static constexpr std::int64_t kPriceScale = 10'000;

    static PriceTick toTicks(double price) noexcept;
    static double fromTicks(PriceTick ticks) noexcept;

    PriceLevel* levelBySide(Side side, PriceTick price);
    const PriceLevel* bestLevel(Side side) const;

    BidBook bids_;
    AskBook asks_;
    OrderPool pool_;
    LocatorTable locator_table_;
};

template <typename Compare>
inline typename OrderBook::PriceLevel& OrderBook::FlatBook<Compare>::levelFor(
    PriceTick price
) {
    auto it = index_.find(price);
    if (it != index_.end()) {
        return *entries_[it->second].level;
    }
    Entry entry{price, std::make_unique<PriceLevel>()};
    const std::size_t pos = findInsertPos(price);
    entries_.insert(entries_.begin() + static_cast<std::ptrdiff_t>(pos), std::move(entry));
    refreshIndices(pos);
    return *entries_[pos].level;
}

template <typename Compare>
inline typename OrderBook::PriceLevel* OrderBook::FlatBook<Compare>::levelOf(
    PriceTick price
) {
    auto it = index_.find(price);
    if (it == index_.end()) {
        return nullptr;
    }
    return entries_[it->second].level.get();
}

template <typename Compare>
inline const OrderBook::PriceLevel* OrderBook::FlatBook<Compare>::levelOf(
    PriceTick price
) const {
    auto it = index_.find(price);
    if (it == index_.end()) {
        return nullptr;
    }
    return entries_[it->second].level.get();
}

template <typename Compare>
inline const OrderBook::PriceLevel* OrderBook::FlatBook<Compare>::bestLevel() const {
    if (entries_.empty()) {
        return nullptr;
    }
    return entries_.front().level.get();
}

template <typename Compare>
inline bool OrderBook::FlatBook<Compare>::eraseIfEmpty(PriceTick price) {
    auto it = index_.find(price);
    if (it == index_.end()) {
        return false;
    }
    const std::size_t idx = it->second;
    if (!entries_[idx].level->empty()) {
        return false;
    }
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(idx));
    index_.erase(it);
    refreshIndices(idx);
    return true;
}

template <typename Compare>
inline std::size_t OrderBook::FlatBook<Compare>::findInsertPos(PriceTick price) const {
    std::size_t pos = 0;
    while (pos < entries_.size() && comp_(entries_[pos].price, price)) {
        ++pos;
    }
    return pos;
}

template <typename Compare>
inline void OrderBook::FlatBook<Compare>::refreshIndices(std::size_t start) {
    for (std::size_t idx = start; idx < entries_.size(); ++idx) {
        index_[entries_[idx].price] = idx;
    }
}

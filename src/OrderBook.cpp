#include "OrderBook.hpp"

#include "perf/Profiler.hpp"

#include <algorithm>  // std::min
#include <cmath>      // std::llround
#include <iostream>
#include <utility>    // std::move

namespace {
constexpr std::size_t kDefaultPoolBlock = 1024;
}

OrderBook::OrderPool::OrderPool(std::size_t blockSize)
    : blockSize_(blockSize ? blockSize : kDefaultPoolBlock) {
    expand();
}

void OrderBook::OrderPool::expand() {
    auto block = std::make_unique<OrderNode[]>(blockSize_);
    OrderNode* raw = block.get();
    for (std::size_t idx = 0; idx < blockSize_; ++idx) {
        freeList_.push_back(&raw[idx]);
    }
    blocks_.push_back(std::move(block));
}

OrderBook::OrderNode* OrderBook::OrderPool::acquire(const Order& order) {
    if (freeList_.empty()) {
        expand();
    }
    OrderNode* node = freeList_.back();
    freeList_.pop_back();
    node->data = order;
    node->next = nullptr;
    node->prev = nullptr;
    return node;
}

void OrderBook::OrderPool::release(OrderNode* node) {
    if (!node) {
        return;
    }
    node->next = nullptr;
    node->prev = nullptr;
    freeList_.push_back(node);
}

OrderBook::PriceTick OrderBook::toTicks(double price) noexcept {
    return static_cast<PriceTick>(std::llround(price * static_cast<double>(kPriceScale)));
}

double OrderBook::fromTicks(PriceTick ticks) noexcept {
    return static_cast<double>(ticks) / static_cast<double>(kPriceScale);
}

OrderBook::PriceLevel* OrderBook::levelBySide(Side side, PriceTick price) {
    return side == Side::Buy ? bids_.levelOf(price) : asks_.levelOf(price);
}

const OrderBook::PriceLevel* OrderBook::bestLevel(Side side) const {
    return side == Side::Buy ? bids_.bestLevel() : asks_.bestLevel();
}

void OrderBook::addLimitOrder(const Order& order) {
    const auto price = toTicks(order.price);
    OrderNode* node = pool_.acquire(order);
    PriceLevel& level =
        order.side == Side::Buy ? bids_.levelFor(price) : asks_.levelFor(price);
    level.append(node);
    locator_table_.set(order.id, Locator{order.side, price, node});
}

bool OrderBook::cancel(OrderId id) {
    Locator* locator = locator_table_.find(id);
    if (locator == nullptr) {
        return false;
    }

    PriceLevel* level = levelBySide(locator->side, locator->price);
    if (level == nullptr) {
        return false;
    }

    level->remove(locator->node);
    pool_.release(locator->node);

    if (locator->side == Side::Buy) {
        bids_.eraseIfEmpty(locator->price);
    } else {
        asks_.eraseIfEmpty(locator->price);
    }

    return locator_table_.erase(id);
}

std::optional<Order> OrderBook::bestBid() const {
    const auto* level = bestLevel(Side::Buy);
    if (level == nullptr || level->head == nullptr) {
        return std::nullopt;
    }
    return level->head->data;
}

std::optional<Order> OrderBook::bestAsk() const {
    const auto* level = bestLevel(Side::Sell);
    if (level == nullptr || level->head == nullptr) {
        return std::nullopt;
    }
    return level->head->data;
}

std::vector<OrderBook::Fill> OrderBook::match(
    Side aggressor,
    double price,
    std::int32_t quantity
) {
    HFT_PROFILE_SCOPE("OrderBook::match");
    std::vector<Fill> fills;
    if (quantity <= 0) {
        return fills;
    }

    PriceTick limit = price > 0.0 ? toTicks(price) : PriceTick{0};
    std::int64_t remaining = quantity;

    auto process_book = [&](auto& book_ref, bool match_asks) {
        auto& entries = book_ref.entries();
        std::size_t idx = 0;

        while (idx < entries.size() && remaining > 0) {
            const PriceTick levelPrice = entries[idx].price;
            bool price_accept = (limit == 0);
            if (!price_accept) {
                if (match_asks) {
                    price_accept = levelPrice <= limit;
                } else {
                    price_accept = levelPrice >= limit;
                }
            }
            if (!price_accept) {
                break;
            }

            PriceLevel& level = *entries[idx].level;
            const double fill_price = fromTicks(levelPrice);
            OrderNode* node = level.head;

            while (node != nullptr && remaining > 0) {
                OrderNode* next = node->next;
                const std::int32_t available = node->data.quantity;
                const std::int32_t executed = static_cast<std::int32_t>(
                    std::min<std::int64_t>(available, remaining)
                );
                if (executed <= 0) {
                    node = next;
                    continue;
                }

                remaining -= executed;
                level.totalQuantity -= executed;
                if (level.totalQuantity < 0) level.totalQuantity = 0;

                const std::int32_t remaining_after = available - executed;
                Fill fill;
                fill.order = node->data;
                fill.order.quantity = remaining_after;
                fill.order.execution_price = fill_price;
                fill.executedQuantity = executed;
                fill.fillPrice = fill_price;
                fills.push_back(fill);

                node->data.quantity = remaining_after;
                if (remaining_after <= 0) {
                    locator_table_.erase(fill.order.id);
                    level.remove(node);
                    pool_.release(node);
                }

                node = next;
            }

            if (level.empty()) {
                book_ref.eraseIfEmpty(levelPrice);
            } else {
                ++idx;
            }
        }
    };

    if (aggressor == Side::Buy) {
        process_book(asks_, true);
    } else {
        process_book(bids_, false);
    }

    return fills;
}

std::vector<OrderBook::LevelSnapshot> OrderBook::levels(std::size_t depth) const {
    std::vector<LevelSnapshot> snapshot;
    snapshot.reserve(depth * 2);

    auto collect = [&](const auto& entries, Side side) {
        std::size_t count = 0;
        for (const auto& entry : entries) {
            if (count++ == depth) {
                break;
            }
            const PriceLevel& level = *entry.level;
            if (level.empty()) {
                continue;
            }
            LevelSnapshot snap;
            snap.price = fromTicks(entry.price);
            snap.totalQuantity = level.totalQuantity;
            snap.side = side;
            for (OrderNode* node = level.head; node != nullptr; node = node->next) {
                snap.resting.push_back(node->data);
            }
            snapshot.push_back(std::move(snap));
        }
    };

    collect(bids_.entries(), Side::Buy);
    collect(asks_.entries(), Side::Sell);
    return snapshot;
}

void OrderBook::printTopLevels(std::size_t depth) const {
    for (const auto& level : levels(depth)) {
        std::cout << (level.side == Side::Buy ? "BID " : "ASK ")
                  << level.price << " qty=" << level.totalQuantity << '\n';
    }
}
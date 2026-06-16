#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "OrderBook.hpp"

TEST_CASE("best bid/ask reflect inserted orders") {
    OrderBook book;
    book.addLimitOrder({1, Side::Buy, 100.10, 5, 10});
    book.addLimitOrder({2, Side::Buy, 100.20, 7, 11});
    book.addLimitOrder({3, Side::Sell, 100.30, 4, 12});
    book.addLimitOrder({4, Side::Sell, 100.40, 6, 13});

    auto bestBid = book.bestBid();
    REQUIRE(bestBid);
    CHECK(bestBid->id == 2);
    CHECK(bestBid->price == Catch::Approx(100.20));
    CHECK(bestBid->quantity == 7);

    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 3);
    CHECK(bestAsk->price == Catch::Approx(100.30));
    CHECK(bestAsk->quantity == 4);
}

TEST_CASE("orders at same price honor FIFO") {
    OrderBook book;
    book.addLimitOrder({10, Side::Buy, 100.00, 5, 100});
    book.addLimitOrder({11, Side::Buy, 100.00, 7, 101});

    auto first = book.bestBid();
    REQUIRE(first);
    CHECK(first->id == 10);

    REQUIRE(book.cancel(10));
    auto second = book.bestBid();
    REQUIRE(second);
    CHECK(second->id == 11);
}

TEST_CASE("cancel removes top-of-book order") {
    OrderBook book;
    book.addLimitOrder({20, Side::Sell, 101.00, 3, 200});
    book.addLimitOrder({21, Side::Sell, 101.10, 4, 201});

    REQUIRE(book.cancel(20));
    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 21);
    CHECK_FALSE(book.cancel(42));
}

TEST_CASE("aggressive limit order executes at resting price") {
    OrderBook book;
    book.addLimitOrder({30, Side::Sell, 100.50, 10, 300});

    auto fills = book.match(Side::Buy, 101.00, 4);

    REQUIRE(fills.size() == 1);
    CHECK(fills[0].order.id == 30);
    CHECK(fills[0].executedQuantity == 4);
    CHECK(fills[0].fillPrice == Catch::Approx(100.50));
    CHECK(fills[0].order.execution_price == Catch::Approx(100.50));

    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 30);
    CHECK(bestAsk->quantity == 6);
}

TEST_CASE("matching consumes multiple price levels without skipping after erase") {
    OrderBook book;
    book.addLimitOrder({40, Side::Sell, 100.50, 4, 400});
    book.addLimitOrder({41, Side::Sell, 100.70, 6, 401});
    book.addLimitOrder({42, Side::Sell, 100.90, 8, 402});

    auto fills = book.match(Side::Buy, 0.0, 7);

    REQUIRE(fills.size() == 2);
    CHECK(fills[0].order.id == 40);
    CHECK(fills[0].executedQuantity == 4);
    CHECK(fills[0].fillPrice == Catch::Approx(100.50));
    CHECK(fills[1].order.id == 41);
    CHECK(fills[1].executedQuantity == 3);
    CHECK(fills[1].fillPrice == Catch::Approx(100.70));

    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 41);
    CHECK(bestAsk->quantity == 3);
}

TEST_CASE("cancel removes non-top FIFO order by locator") {
    OrderBook book;
    book.addLimitOrder({50, Side::Sell, 101.00, 3, 500});
    book.addLimitOrder({51, Side::Sell, 101.00, 4, 501});

    REQUIRE(book.cancel(51));

    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 50);
    CHECK(bestAsk->quantity == 3);

    auto fills = book.match(Side::Buy, 101.00, 10);
    REQUIRE(fills.size() == 1);
    CHECK(fills[0].order.id == 50);
    CHECK(fills[0].executedQuantity == 3);
    CHECK_FALSE(book.bestAsk());
}

TEST_CASE("limit price prevents crossing beyond acceptable levels") {
    OrderBook book;
    book.addLimitOrder({60, Side::Sell, 100.50, 4, 600});
    book.addLimitOrder({61, Side::Sell, 100.70, 6, 601});

    auto fills = book.match(Side::Buy, 100.60, 10);

    REQUIRE(fills.size() == 1);
    CHECK(fills[0].order.id == 60);
    CHECK(fills[0].executedQuantity == 4);

    auto bestAsk = book.bestAsk();
    REQUIRE(bestAsk);
    CHECK(bestAsk->id == 61);
    CHECK(bestAsk->quantity == 6);
}
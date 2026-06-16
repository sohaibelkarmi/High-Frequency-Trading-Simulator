cmake_minimum_required(VERSION 3.15)
project(order_book_tests LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Catch2 3 REQUIRED)

add_executable(order_tests
  ${CMAKE_CURRENT_LIST_DIR}/../../order_tests.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../../src/OrderBook.cpp
)

target_include_directories(order_tests PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../../src
)

target_compile_options(order_tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(order_tests PRIVATE Catch2::Catch2WithMain)

add_test(NAME order_tests COMMAND order_tests)

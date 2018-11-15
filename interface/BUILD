cc_library(
  name = "function",
  hdrs = ["function.hh"],
  deps = [
       "@eigen//:eigen",
  ],
)

cc_test(
  name = "function_test",
  srcs = ["function_test.cc"],
  deps = [
    ":function",
    "@gtest//:gtest_main",
  ]
)

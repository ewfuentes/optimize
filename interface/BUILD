

cc_library(
  name = "function",
  hdrs = ["function.hh"],
  deps = [
       "@eigen//:eigen",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "function_test",
  srcs = ["function_test.cc"],
  deps = [
    ":function",
    "@gtest//:gtest_main",
  ]
)

cc_library(
  name = "optimizer",
  srcs = ["optimizer.hh"],
  deps = [
    ":function",
  ],
  visibility = ["//visibility:public"],
)

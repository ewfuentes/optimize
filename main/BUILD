cc_binary(
  name = "hello_world",
  srcs = ["hello_world.cc"],
)

cc_test(
  name = "hello_test",
  srcs = ["hello_test.cc"],
  deps = [
    "@gtest//:gtest_main",
    "@eigen//:eigen",
    "@glog//:glog",
    ]
)
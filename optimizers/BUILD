cc_library(
  name = "optimizer",
  hdrs = ["optimizer.hh"],
  deps = [ "@eigen//:eigen"]
  )

cc_library(
  name = "simple_gradient_descent",
  hdrs = ["simple_gradient_descent.hh"],
  srcs = ["simple_gradient_descent.cc"],
  deps = [
    ":optimizer",
  ]
)

cc_test(
  name = "simple_gradient_descent_test",
  srcs = ["simple_gradient_descent_test.cc"],
  deps = [
   ":simple_gradient_descent",
   ":optimizer",
   "@gtest//:gtest_main",
   ]
 )

cc_library(
  name = "simple_newtons_method",
  hdrs = ["simple_newtons_method.hh"],
  srcs = ["simple_newtons_method.cc"],
  deps = [
    ":optimizer",
  ]
)

cc_test(
  name = "simple_newtons_method_test",
  srcs = ["simple_newtons_method_test.cc"],
  deps = [
   ":simple_newtons_method",
   ":optimizer",
   "@gtest//:gtest_main",
   ]
 )
cc_library(
  name = "optimizer",
  hdrs = ["optimizer.hh"],
  deps = [ "@eigen//:eigen"]
)

cc_library(
  name = "armijo_line_search",
  hdrs = ["armijo_line_search.hh"],
  srcs = ["armijo_line_search.cc"],
  deps = [
    ":optimizer",
  ]
)

cc_test(
  name = "armijo_line_search_test",
  srcs = ["armijo_line_search_test.cc"],
  deps = [
    ":armijo_line_search",
    "@gtest//:gtest_main",
  ]
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

cc_library(
  name = "constraint",
  hdrs = ["constraint.hh"],
  deps = [
  ":optimizer",
  ]
)

cc_library(
  name = "constrained_newtons_method",
  hdrs = ["constrained_newtons_method.hh"],
  srcs = ["constrained_newtons_method.cc"],
  deps = [
    ":optimizer",
    ":constraint",
    ":simple_newtons_method",
  ]
)

cc_test(
  name = "constrained_newtons_method_test",
  srcs = ["constrained_newtons_method_test.cc"],
  deps = [
   ":constrained_newtons_method",
   ":optimizer",
   "@gtest//:gtest_main",
   ]
 )
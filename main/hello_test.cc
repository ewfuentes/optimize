#include <Eigen/Dense>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>

TEST(HelloTest, hello_test) {
  google::InitGoogleLogging("");
  std::cout << "Hello World!" << std::endl;
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  LOG(ERROR) << x.transpose() << std::endl;
  EXPECT_TRUE(true);
}

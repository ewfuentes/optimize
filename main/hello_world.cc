#include <iostream>
#include <Eigen/Dense>
#include <glog/logging.g>

int main() {
  std::cout << "Hello World!" << std::endl;
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  LOG(ERROR) << x.transpose() << std::endl;

}

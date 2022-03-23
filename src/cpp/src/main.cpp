#include "maddness.hpp"
#include <stdio.h>

int main(int argc, char *const argv[]) {
  printf("Test\n");

  maddness_amm_task<float> task(64, 32, 4, 64, 4);
  task.run_matmul();

  std::cout << task.X;
  std::cout << task.Q << "\n";
  std::cout << task.output();
}

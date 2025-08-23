#include <sycl/sycl.hpp>
#include <iostream>

int main()
{
  // CPU choice
  sycl::queue q1(sycl::cpu_selector_v);
  std::cout << q1.get_device().get_info<sycl::info::device::name>() << "\n";

  // GPU choice with checking
  try
  {
    sycl::queue q2(sycl::gpu_selector_v);
    std::cout << q2.get_device().get_info<sycl::info::device::name>() << "\n";
  }
  catch(sycl::exception& e)
  {
    std::cout << "No GPU available !" << "\n";
  }
}
#include <sycl/sycl.hpp>
#include <iostream>

int main()
{
  sycl::queue q;
  std::cout << std::boolalpha << q.get_device().get_info<sycl::info::device::name>()  << "\n";
  std::cout << std::boolalpha << q.get_device().has(sycl::aspect::cpu)                << "\n";
  std::cout << std::boolalpha << q.get_device().has(sycl::aspect::gpu)                << "\n";
  std::cout << std::boolalpha << q.get_device().has(sycl::aspect::fp16)               << "\n";
  std::cout << std::boolalpha << q.get_device().has(sycl::aspect::fp64)               << "\n";
  std::cout << std::boolalpha << q.get_device().has(sycl::aspect::atomic64)           << "\n";
  std::cout << std::boolalpha << q.get_device().has(sycl::aspect::queue_profiling)    << "\n";
}
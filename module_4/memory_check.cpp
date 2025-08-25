#include <sycl/sycl.hpp>

int main()
{
  sycl::queue Q;

  bool shared = Q.get_device().has(sycl::aspect::usm_shared_allocations);
  bool device = Q.get_device().has(sycl::aspect::usm_device_allocations);

  std::cout << "Support:\n";
  std::cout << "  shared memory - " << std::boolalpha << shared << "\n";
  std::cout << "  device memory - " << std::boolalpha << device << "\n";
}
#include <sycl/sycl.hpp>
#include <string>

struct device_info {
    std::string name;
    std::string vendor;
    bool gpu;
    
    // Constructor initializes members from sycl::device
    device_info(const sycl::device& dev)
        : name(dev.get_info<sycl::info::device::name>()),
          vendor(dev.get_info<sycl::info::device::vendor>()),
          gpu(dev.is_gpu())
    {}
};

void exercice()
{
  sycl::queue q;
  auto[name, vendor, status] = device_info(q.get_device());
  
  std::cout << "Device Name     : " << name                     << std::endl;
  std::cout << "Device Vendor   : " << vendor                   << std::endl;
  std::cout << "Device is a GPU : " << std::boolalpha << status << std::endl;
}

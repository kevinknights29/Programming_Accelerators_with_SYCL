#include <sycl/sycl.hpp>

sycl::queue get_cpu()
{
  return sycl::queue{sycl::aspect_selector(sycl::aspect::cpu)};
}

sycl::queue get_gpu()
{
  return sycl::queue{sycl::aspect_selector(sycl::aspect::gpu)};
}

sycl::queue get_gpu_or_default(const char* vendor)
{
  auto selector = [vendor](const sycl::device& d) {
    // Check if device is a GPU, and vendor matches
    bool is_gpu = d.is_gpu();
    bool vendor_match = d.get_info<sycl::info::device::vendor>().find(vendor) != std::string::npos;
    return (is_gpu && vendor_match) ? 1 : -1; // 1 to select, -1 to ignore
  };
  try {
    // Returns queue for device with matching vendor, or throws if not found
    return sycl::queue(selector);
  } catch (...) {
    // Fallback: return default device
    return sycl::queue(sycl::default_selector_v);
  }
}

sycl::queue get_gpu_or_cpu(const char* vendor)
{
  auto selector = [vendor](const sycl::device& d) {
    // Check if device is a GPU, and vendor matches
    bool is_gpu = d.is_gpu();
    bool vendor_match = d.get_info<sycl::info::device::vendor>().find(vendor) != std::string::npos;
    return (is_gpu && vendor_match) ? 1 : -1; // 1 to select, -1 to ignore
  };
  try {
    // Returns queue for device with matching vendor, or throws if not found
    return sycl::queue(selector);
  } catch (...) {
    // Fallback: return default device
    return sycl::queue(sycl::aspect_selector(sycl::aspect::cpu));
  }
}

sycl::queue get_gpu_or_fail(const char* vendor)
{
  return sycl::queue([vendor](const sycl::device& d) {
    // Check if device is a GPU, and vendor matches
    bool is_gpu = d.is_gpu();
    bool vendor_match = d.get_info<sycl::info::device::vendor>().find(vendor) != std::string::npos;
    return (is_gpu && vendor_match) ? 1 : -1; // 1 to select, -1 to ignore
  });
}

void exercice()
{
  {
    auto q = get_cpu();
    std::cout << "get_cpu() : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  }

  {
    auto q = get_gpu();
    std::cout << "get_gpu() : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  }

  {
    auto q = get_gpu_or_default("NVIDIA Corporation");
    std::cout << "get_gpu_or_default(\"NVIDIA Corporation\") : " 
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
  }

  {
    auto q = get_gpu_or_cpu("NVIDIA Corporation");
    std::cout << "get_gpu_or_cpu(\"NVIDIA Corporation\") : " 
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
  }

  {
    auto q = get_gpu_or_cpu("ACME LTD");
    std::cout << "get_gpu_or_cpu(\"ACME LTD\") : " 
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
  }

  {
    auto q = get_gpu_or_fail("NVIDIA Corporation");
    std::cout << "get_gpu_or_fail(\"NVIDIA Corporation\") : " 
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
  }

  try
  {
    auto q = get_gpu_or_fail("ACME LTD");
    std::cout << "get_gpu_or_fail(\"ACME LTD\") : " 
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
  }
  catch(sycl::exception& e)
  {
    std::cout << "Expected error: " << e.what() << "\n";
  }
}
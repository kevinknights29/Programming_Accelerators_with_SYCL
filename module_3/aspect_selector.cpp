#include <sycl/sycl.hpp>
#include <iostream>

int main()
{
    sycl::queue q0{sycl::aspect_selector()};
    std::cout << q0.get_device().get_info<sycl::info::device::name>() << "\n";

    sycl::queue q1{sycl::aspect_selector(sycl::aspect::gpu, sycl::aspect::fp16)};
    std::cout << q1.get_device().get_info<sycl::info::device::name>() << "\n";

    // Using vectors
    sycl::queue q0{sycl::aspect_selector(std::vector{sycl::aspect::fp16, sycl::aspect::gpu})};
    std::cout << q0.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    sycl::queue q1{sycl::aspect_selector(std::vector{sycl::aspect::cpu}, std::vector{sycl::aspect::accelerator})};
    std::cout << q1.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Custom Selector
    int is_it_intel(sycl::device const &d)
    {
        return d.get_info<sycl::info::device::vendor>().starts_with("Intel");
    }

    //Select an Intel device or try again.
    sycl::queue q0{is_it_intel};
    std::cout << q0.get_device().get_info<sycl::info::device::name>() << "\n";

    // Checks for half type or fails.
    try
    {
        sycl::queue q1{[](auto &d)
                       { return d.has(sycl::aspect::fp16) ? 1 : -1; }};
        std::cout << q1.get_device().get_info<sycl::info::device::name>() << "\n";
    }
    catch (sycl::exception &e)
    {
        std::cout << "No device available !" << "\n";
    }
}
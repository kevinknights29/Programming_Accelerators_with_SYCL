#include <algorithm>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp> // SYCL header

int main()
{
  // (1) Data to be handled
  std::vector<float> data = { 1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f
                            , 9.f,10.f,11.f,12.f,13.f,14.f,15.f,16.f
                            };

  // (2) Operation queue for our default accelerator
  sycl::queue Q;

  {
    // (3) Making our data available to the accelerator
    sycl::buffer device_data(data);

    // (4) Submitting an Action to the Device
    Q.submit([&](sycl::handler& h)
    {
        // (5) Parallel execution of the computation kernel
        sycl::accessor acc{device_data,h};
        h.parallel_for(data.size(), [=](auto i) { acc[i] = 1.f/acc[i]; } );
    });
  }

  // (6) Displaying results
  for (int i = 0; i < data.size(); i++)
    std::cout << "data[" << i << "] = " << data[i] << "\n";
}
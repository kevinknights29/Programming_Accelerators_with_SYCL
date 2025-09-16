#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <limits>

int spread(sycl::queue& q, sycl::buffer<int> values)
{
    auto size = values.get_range().size();
    
    // Create buffers for min and max results
    int min_result = std::numeric_limits<int>::max();
    int max_result = std::numeric_limits<int>::min();
    sycl::buffer<int> min_buffer{&min_result, 1};
    sycl::buffer<int> max_buffer{&max_result, 1};
    
    q.submit([&](sycl::handler& h) {
        sycl::accessor values_acc{values, h, sycl::read_only};
        
        // Find minimum using reduction
        h.parallel_for(sycl::range<1>(size),
            sycl::reduction(min_buffer, h, std::numeric_limits<int>::max(), sycl::minimum<int>{}),
            [=](sycl::id<1> idx, auto& min_reducer) {
                min_reducer.combine(values_acc[idx]);
            });
    });
    
    q.submit([&](sycl::handler& h) {
        sycl::accessor values_acc{values, h, sycl::read_only};
        
        // Find maximum using reduction  
        h.parallel_for(sycl::range<1>(size),
            sycl::reduction(max_buffer, h, std::numeric_limits<int>::min(), sycl::maximum<int>{}),
            [=](sycl::id<1> idx, auto& max_reducer) {
                max_reducer.combine(values_acc[idx]);
            });
    });
    
    // Wait for both operations to complete
    q.wait();
    
    // Get results from host
    auto min_host = min_buffer.get_host_access();
    auto max_host = max_buffer.get_host_access();
    
    return max_host[0] - min_host[0]; // max - min
}

void try_it_out()
{
  std::vector<int> data = {-3, 7, -8, 5, 2, -9, 10, 0, -6, 4};
  sycl::queue  q;
  sycl::buffer values{ data };

  std::cout << "Data set: ";
  for(auto e : data)
      std::cout << e << " ";
  std::cout << std::endl;

  std::cout << "SPREAD : " << spread(q,values) << " vs " << 19 << std::endl;
}
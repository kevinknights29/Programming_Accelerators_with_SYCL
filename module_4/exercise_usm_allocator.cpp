#include <sycl/sycl.hpp>

using alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;

void print(char const* name, auto const& arr)
{
  std::cout << name << ": ";
  for(auto e : arr) std::cout << e << " ";
  std::cout << std::endl;
}

void process(sycl::queue& q, std::vector<int, alloc_t>& arr)
{
  q.parallel_for( arr.size()
                , [px = arr.data()](auto i) { px[i] = i * i; }
                );
  q.wait();
}

void exercice(sycl::queue& Q)
{
  std::size_t size = 10;
  
  // Define V, a vector of int containing size element, using an USM allcoator.
  std::vector<int, alloc_t> V(size, alloc_t{Q}); 

  process(Q,V);

  print("V", V);
}

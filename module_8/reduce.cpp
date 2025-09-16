#include <vector>
#include <iostream>
#include <sycl/sycl.hpp>

int main()
{
  constexpr std::size_t size = 16;
  int result = 0;

  std::vector<int> data(size);
  for(std::size_t i=0;i<size;++i)
    data[i] = i+1;

  sycl::queue q;
  {
    sycl::buffer      values{ data };
    sycl::buffer<int> sum(&result, 1);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor v{values, h, sycl::read_only};
      auto reductor = sycl::reduction(sum, h, sycl::plus<>());

      h.parallel_for( sycl::range{size}, reductor
                    , [=](auto i, auto& acc) { acc += v[i]; }
                    );
    });
  }

  std::cout << result << "\n";

  return 0;
}
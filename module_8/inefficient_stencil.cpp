#include <iostream>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>

int main()
{
  constexpr std::size_t size = 15;

  std::vector<int>    data(size);
  std::vector<float>  results(size, 0.f);

  std::random_device  dev;
  std::mt19937        eng{dev()};
  std::uniform_int_distribution dist {1,10};
  std::generate(data.begin(), data.end(), [&]() { return dist(eng); });

  for(auto e : data)
    std::cout << e << " ";
  std::cout << "\n";

  sycl::queue q;
  {
    sycl::buffer values(data);
    sycl::buffer outs(results);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor in{values, h, sycl::read_only};
      sycl::accessor out{outs, h};

      h.parallel_for( sycl::range{size}
                    , [=](auto i)
                      {
                        if(i >= 1 && i < size-1)
                          out[i] = (in[i-1] + in[i] + in[i+1])/3.f;
                      }
                    );
    });
  }


  for(auto e : results)
    std::cout << e << " ";
  std::cout << "\n";
  return 0;
}
// REFERENCE: 
// namespace sycl
// {
//   template<typename T = void> struct plus;
//   template<typename T = void> struct multiplies;
//   template<typename T = void> struct bit_and;
//   template<typename T = void> struct bit_or;
//   template<typename T = void> struct bit_xor;
//   template<typename T = void> struct logical_and;
//   template<typename T = void> struct logical_or;
//   template<typename T = void> struct minimum;
//   template<typename T = void> struct maximum;
// }

#include <iostream>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>

int main()
{
  constexpr std::size_t size = 16;
  int result = 0;

  std::vector<int> data(size);
  std::random_device  dev;
  std::mt19937        eng{dev()};
  std::uniform_int_distribution dist {-50,50};
  std::generate(data.begin(), data.end(), [&]() { return dist(eng); });

  for(auto e : data)
    std::cout << e << " ";
  std::cout << "\n";

  sycl::queue q;
  {
    sycl::buffer      values{ data };
    sycl::buffer<int> mx(&result, 1);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor v{values, h, sycl::read_only};
      auto reductor = sycl::reduction(mx, h, sycl::maximum<>());

      h.parallel_for( sycl::range{size}, reductor
                    , [=](auto i, auto& acc) { acc.combine(v[i]); }
                    );
    });
  }

  std::cout << result << "\n";

  return 0;
}
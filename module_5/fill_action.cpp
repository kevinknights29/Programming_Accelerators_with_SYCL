#include <sycl/sycl.hpp>

int main()
{
  sycl::queue Q;

  constexpr auto sz = 10;
  auto dst = sycl::malloc_shared<int>(sz, Q);

  // Sets dst to value 42
  Q.fill(dst, 42, sz);
  Q.wait();

  for(int i = 0;i<sz; ++i)
      std::cout << dst[i] << " ";
  std::cout << '\n';

  // Handler version
  std::vector<int> dst(10);

  {
    sycl::buffer b{ dst };

    Q.submit([&](sycl::handler& h)
    {
      sycl::accessor out{b, h, sycl::write_only };

      // Sets dst to value 42
      h.fill(out, 63);
    });
  }

  for(auto e: dst)
    std::cout << e << " ";
  std::cout << '\n';
}
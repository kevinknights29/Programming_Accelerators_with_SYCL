#include <sycl/sycl.hpp>

int main()
{
  sycl::queue Q;

  constexpr auto sz = 10;
  auto src = sycl::malloc_shared<int>(sz, Q);
  auto dst = sycl::malloc_shared<int>(sz/2, Q);
  for(int i=0;i<sz;++i)
      src[i] = 1 + i;

  // Copy the first half of src into dst
  Q.copy(src, dst, sz/2);
  Q.wait();

  for(int i = 0;i<sz/2; ++i)
      std::cout << dst[i] << " ";
  std::cout << '\n';

  // Handler version:
  std::vector<int> src{1,2,3,4,5,6,7,8,9,10};
  std::vector<int> dst(5);

  {
    sycl::buffer b0{ src.data(), sycl::range{sz/2} };
    sycl::buffer b1{ dst };

    Q.submit([&](sycl::handler& h)
    {
      sycl::accessor in {b0, h, sycl::read_only  };
      sycl::accessor out{b1, h, sycl::write_only };

      // Copie la 1e moitié de src dans dst
      h.copy(in, out);
    });
  }

  for(auto e: dst)
    std::cout << e << " ";
  std::cout << '\n';
}
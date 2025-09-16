#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

template<typename Function, typename In, typename Out>
void map(sycl::queue& q, Function F, In const& x, Out& y)
{
  sycl::buffer bx(x);
  sycl::buffer by(y);

  q.submit([&](sycl::handler& h)
  {
    sycl::accessor ax{bx, h, sycl::read_only};
    sycl::accessor ay{by, h};
    h.parallel_for( bx.size(), [=](auto i) { ay[i] = F(ax[i]); } );
  });
}

int main()
{
  sycl::queue q;
  std::size_t size = 16;

  std::vector<float> X(size);
  std::vector<float> Y(size, 0.f);

  for(std::size_t i = 0;i < size;++i)
    X[i] = 1.f + i;

  for(auto e : X)
    std::cout << e << " ";
  std::cout << "\n";

  map(q, [](auto e) { return sqrt(e); }, X, Y);

  for(auto e : Y)
    std::cout << e << " ";
  std::cout << "\n";

  return 0;
}
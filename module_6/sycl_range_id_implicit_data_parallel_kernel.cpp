#include <sycl/sycl.hpp>
#include <vector>

int main()
{
  sycl::queue Q;
  std::size_t size = 8;

  std::vector<float> X_h(size);
  std::vector<float> Y_h(size);
  float a = 1.25f;

  for(std::size_t i = 0;i < size;++i)
  {
    X_h[i] = 1.f + i;
    Y_h[i] = 2.5f * i;
  }

  {
    sycl::buffer<float> X_d(X_h);
    sycl::buffer<float> Y_d(Y_h);

    Q.submit([&](sycl::handler& h)
    {
      sycl::accessor x{X_d, h};
      sycl::accessor y{Y_d, h};
      h.parallel_for( sycl::range<1>(size), [=](sycl::id<1> i) { return y[i] += x[i]; } );
    });
  }

  for(std::size_t i = 0;i < size;++i)
  {
    std::cout << Y_h[i] << " ";
  }
  std::cout << std::endl;
}
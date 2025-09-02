#include <sycl/sycl.hpp>

int main()
{
  sycl::queue Q;

  constexpr auto sz = 10;
  float* X = sycl::malloc_shared<float>(sz, Q);
  float* Y = sycl::malloc_shared<float>(sz, Q);

  float A = 3.14f;

  auto ex = Q.fill(X,  2.f, sz);
  auto ey = Q.fill(Y, 10.f, sz);

  Q.parallel_for( sz, {ex, ey}, [=](auto i) { Y[i] += A * X[i];} );
  Q.wait();

  for(int i=0; i < sz; ++i)
    std::cout << Y[i] << " ";
  std::cout << '\n';
}
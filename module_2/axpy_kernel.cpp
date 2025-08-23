#include <sycl/sycl.hpp>

int main()
{
  sycl::queue Q;

  std::size_t size = 8;
  float *X = sycl::malloc_shared<float>(size, Q);
  float *Y = sycl::malloc_shared<float>(size, Q);

  for(std::size_t i = 0;i < size;++i)
  {
    X[i] = 1.f + i;
    Y[i] = 2.5f * i;
  }

  float a = 1.25f;

  Q.parallel_for(size, [a,X,Y](auto i) { return Y[i] += a*X[i]; });

  // Ensure execution completes before reading results
  Q.wait();

  for(std::size_t i = 0;i < size;++i)
  {
    std::cout << Y[i] << " ";
  }
  std::cout << std::endl;

  sycl::free(X, Q);
  sycl::free(Y, Q);
}
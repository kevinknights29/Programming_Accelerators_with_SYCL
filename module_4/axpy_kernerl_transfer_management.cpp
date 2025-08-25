#include <sycl/sycl.hpp>
#include <vector>

int main()
{
  sycl::queue Q;

  std::size_t size = 8;
  std::vector<float> X_h(size);
  std::vector<float> Y_h(size);

  float* X_d = sycl::malloc_device<float>(size, Q);
  float* Y_d = sycl::malloc_device<float>(size, Q);

  for(std::size_t i = 0;i < size;++i)
  {
    X_h[i] = 1.f + i;
    Y_h[i] = 2.5f * i;
  }

  Q.memcpy(X_d, X_h.data(), size * sizeof(float));
  Q.memcpy(Y_d, Y_h.data(), size * sizeof(float));
  Q.wait();

  float a = 1.25f;
  Q.parallel_for(size, [a,X_d,Y_d](auto i) { return Y_d[i] += a*X_d[i]; });
  Q.wait();

  Q.memcpy(Y_h.data(), Y_d, size * sizeof(float));
  Q.wait();

  for(std::size_t i = 0;i < size;++i)
  {
    std::cout << Y_h[i] << " ";
  }
  std::cout << std::endl;

  sycl::free(X_d,Q);
  sycl::free(Y_d,Q);
}
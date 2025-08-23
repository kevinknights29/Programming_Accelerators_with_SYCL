#include <sycl/sycl.hpp>

void print(char const* name, int * arr, int size)
{
  std::cout << name << ": ";
  for(std::size_t i = 0;i < size;++i) std::cout << arr[i] << " ";
  std::cout << std::endl;
}

void reverse(sycl::queue& Q, int* in, int* out, int size)
{
  Q.parallel_for(size, [in, out, size](auto i) { return out[i] = in[(size - 1) - i]; });
  Q.wait();
}

void exercice(sycl::queue& Q)
{
  std::size_t size = 10;
  int *X = sycl::malloc_shared<int>(size, Q);
  int *Y = sycl::malloc_shared<int>(size, Q);

  for(std::size_t i = 0;i < size;++i)
    X[i] = 1+ i;

  reverse(Q,X,Y,size);

  print("X", X, size);
  print("Y", Y, size);

  sycl::free(X,Q);
  sycl::free(Y,Q);
}
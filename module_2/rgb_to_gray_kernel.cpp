#include <sycl/sycl.hpp>

struct pixel
{
  int r,g,b;
};

void print(char const* name, pixel* arr, int size)
{
  std::cout << name << " :";
  for(std::size_t i = 0;i < size;++i) 
    std::cout << "{" << arr[i].r << " " << arr[i].g << " " << arr[i].b << "} ";
  std::cout << std::endl;
}

void to_grey(sycl::queue& Q, pixel* in, pixel* out, int size)
{
  Q.parallel_for(size, [=](auto i) { 
    auto avg = (in[i].r + in[i].g + in[i].b) / 3; 
    out[i] = {avg, avg, avg}; 
  });
  Q.wait();
}

void exercice()
{
  sycl::queue Q;

  int size = 10;
  pixel *X = sycl::malloc_shared<pixel>(size, Q);
  pixel *Y = sycl::malloc_shared<pixel>(size, Q);

  for(int i = 0;i < size;++i)
    X[i] = pixel{i,i+1,255-i*10};

  to_grey(Q, X, Y, size);

  print("X", X, size);
  print("Y", Y, size);

  sycl::free(X,Q);
  sycl::free(Y,Q);
}
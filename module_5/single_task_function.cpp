#include <sycl/sycl.hpp>

int main()
{
  sycl::queue Q;

  // Queue version
  auto ptr = sycl::malloc_shared<float>(10, Q);
  Q.single_task( [=]() { for(int i=0;i<1024;++i) ptr[i] = i/2.f; } );
  Q.wait();

  // Handler version
  {
    sycl::buffer b{ptr, sycl::range{10}};
    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor acc{b, h};
                h.single_task( [=]() { acc[0] = -1.f ;} );
              }
            );
  }

  for(int i=0;i<10;i++)
    std::cout << ptr[i] << " ";
  std::cout << '\n';
}
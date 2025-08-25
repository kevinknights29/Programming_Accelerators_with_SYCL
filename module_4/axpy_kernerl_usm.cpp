#include <sycl/sycl.hpp>
#include <vector>

int main()
{
  sycl::queue Q;

  std::size_t size = 8;
  using alloc_t = sycl::usm_allocator<float,sycl::usm::alloc::shared>;

  std::vector<float,alloc_t> X(size, alloc_t{Q});
  std::vector<float,alloc_t> Y(size, alloc_t{Q});

  for(std::size_t i = 0;i < size;++i)
  {
    X[i] = 1.f + i;
    Y[i] = 2.5f * i;
  }

  float a = 1.25f;

  Q.parallel_for( size
                , [a,px = X.data(), py = Y.data()](auto i)
                  {
                    return py[i] += a*px[i];
                  }
                );
  Q.wait();

  for(std::size_t i = 0;i < size;++i)
  {
    std::cout << Y[i] << " ";
  }
  std::cout << std::endl;
}
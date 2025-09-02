#include <sycl/sycl.hpp>
#include <vector>

int main()
{
  sycl::queue Q;

  std::vector<float> X(10);
  std::vector<float> Y(10);

  {
    sycl::buffer bx{X}, by{Y};
    float A = 3.14f;

    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor acc{bx, h, sycl::write_only};
                h.fill(acc, 2.f);
              }
            );

    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor acc{by, h, sycl::write_only};
                h.fill(acc,10.f);
              }
            );

    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor ax{bx, h, sycl::read_only};
                sycl::accessor ay{by, h};

                h.parallel_for( by.size()
                              , [=](auto i) { ay[i] += A * ax[i];}
                              );
              }
            );
  }

  for(auto e: Y)
    std::cout << e << " ";
  std::cout << '\n';
}
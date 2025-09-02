#include <sycl/sycl.hpp>
#include <vector>


int main()
{
  sycl::queue Q;

  constexpr auto sz = 10;
  std::vector<float> X(sz);
  std::vector<float> Y(sz);

  {
    sycl::buffer bx{X}, by{Y};
    float A = 3.14f;

    auto ex = Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor acc{bx, h};
                h.fill(acc, 2.f);
              }
            );

    auto ey = Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor acc{by, h};
                h.fill(acc,10.f);
              }
            );

    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor ax{bx, h};
                sycl::accessor ay{by, h};

                h.depends_on({ex,ey});
                h.parallel_for( sz
                              , [=](auto i) { ay[i] += A * ax[i];}
                              );
              }
            );
  }

  for(auto e: Y)
    std::cout << e << " ";
  std::cout << '\n';
}
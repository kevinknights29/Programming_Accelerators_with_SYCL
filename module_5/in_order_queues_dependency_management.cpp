#include <sycl/sycl.hpp>
#include <vector>

int main()
{
  sycl::queue Q(sycl::property::queue::in_order{});

  constexpr auto sz = 10;
  std::vector<float> X(sz);
  std::vector<float> Y(sz);

  {
    sycl::buffer bx{X}, by{Y};
    float A = 3.14f;

    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor a{bx, h, sycl::write_only};
                h.fill(a, 2.f);
              }
            );

    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor a{by, h, sycl::read_write};
                h.fill(a,10.f);
              }
            );

    Q.submit( [&](sycl::handler& h)
              {
                sycl::accessor ax{bx, h, sycl::read_only};
                sycl::accessor ay{by, h, sycl::read_write};
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
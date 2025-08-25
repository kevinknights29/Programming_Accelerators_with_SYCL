#include <sycl/sycl.hpp>
#include <vector>

int main()
{
  sycl::queue Q;
  size_t size = 15;
  std::vector<float> X_h(size);
  std::vector<float> Y_h(size);
  float a = 1.25f;

  for (size_t i = 0; i < size; i++)
  {
    X_h[i] = 1.f + i;
    Y_h[i] = 2.5f * i;
  }

  {
    sycl::buffer X_d(X_h);
    sycl::buffer Y_d(Y_h);

    Q.submit([&](sycl::handler& h)
    {
      sycl::accessor<float,1> x{X_d, h};
      sycl::accessor<float,1> y{Y_d, h};

      h.parallel_for( X_d.get_range()
                    , [=](auto i) { 
                        y[i] += a * x[i]; 
                    }
                    );
    });
  }

  for(auto e: Y_h)
    std::cout << e << " ";
  std::cout << "\n";
}
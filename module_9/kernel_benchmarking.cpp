#include <sycl/sycl.hpp>
#include <vector>

int main()
{
  sycl::queue Q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
  constexpr std::size_t r = 20;
  constexpr std::size_t n = r*20;

  std::vector<float> A_h(n*n), B_h(n*n), C_h(n*n);

  {
    sycl::buffer A_d(A_h.data(),sycl::range(n,n));
    sycl::buffer B_d(B_h.data(),sycl::range(n,n));
    sycl::buffer C_d(C_h.data(),sycl::range(n,n));

    double t01 = 0;
    for(int i=0;i<50;++i)
    {
        auto e = Q.submit([&](sycl::handler& h)
        {
            sycl::accessor<float,2> A{A_d, h};
            sycl::accessor<float,2> B{B_d, h};
            sycl::accessor<float,2> C{C_d, h};

            auto global = sycl::range{n,n};
            auto local  = sycl::range{r,r};
            h.parallel_for( sycl::nd_range{global,local}
                            , [=](sycl::nd_item<2> idx)
                            {
                                auto i = idx.get_global_id(0);
                                auto j = idx.get_global_id(1);
                                C[i][j] = A[i][j] * B[i][j];
                            }
                            );
        });
        t01 += (    e.template get_profiling_info<sycl::info::event_profiling::command_end>()
                -   e.template get_profiling_info<sycl::info::event_profiling::command_start>()
                );
    }

    std::cout  << "[0][1]: "  << t01/50. << std::endl;

    double t10 = 0;
    for(int i=0;i<50;++i)
    {
        auto e = Q.submit([&](sycl::handler& h)
        {
            sycl::accessor<float,2> A{A_d, h};
            sycl::accessor<float,2> B{B_d, h};
            sycl::accessor<float,2> C{C_d, h};

            auto global = sycl::range{n,n};
            auto local  = sycl::range{r,r};
            h.parallel_for( sycl::nd_range{global,local}
                            , [=](sycl::nd_item<2> idx)
                            {
                                auto i = idx.get_global_id(1);
                                auto j = idx.get_global_id(0);
                                C[i][j] = A[i][j] * B[i][j];
                            }
                            );
        });
        t10 += (    e.template get_profiling_info<sycl::info::event_profiling::command_end>()
                -   e.template get_profiling_info<sycl::info::event_profiling::command_start>()
                );
    }

    std::cout  << "[1][0]: "  << t10/50. << std::endl;
  }
}
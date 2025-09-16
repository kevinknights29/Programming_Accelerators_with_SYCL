#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

int main()
{
  std::vector<int> data(32768,1);
  std::vector<int> out(512);

  sycl::queue q;
  {
    sycl::buffer values(data);
    sycl::buffer redux(out);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor dv{values, h, sycl::read_only};
      sycl::accessor rd{redux, h};
      sycl::local_accessor<int> cache{data.size()/out.size(),h};

      h.parallel_for( sycl::nd_range{{data.size()},{data.size()/out.size()}}
                    , [=](sycl::nd_item<1> i)
                      {
                        auto lid = i.get_local_id(0);
                        auto gid = i.get_global_id(0);
                        auto rid = i.get_group_linear_id();

                        cache[lid] = dv[gid];
                        sycl::group_barrier(i.get_group());

                        int sum = 0;
                        for(int i=0;i<cache.size();++i)
                          sum += cache[i];

                        rd[rid] = sum;
                      }
                    );
    });
  }

  int total = 0;

  for(auto e : out)
    total += e;

  std::cout << total << std::endl;
  // NOTE: Returns correct result.
}
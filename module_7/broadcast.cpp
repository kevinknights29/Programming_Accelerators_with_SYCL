#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

int main()
{
  std::vector<int> data{1,2,3,4,5,6,7,8,9,9,9,9,10,10,10,10};
  std::vector<int> out(4);

  sycl::queue q;
  {
    sycl::buffer values(data);
    sycl::buffer output(out);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor dv{values, h, sycl::read_only};
      sycl::accessor rd{output, h};

      h.parallel_for( sycl::nd_range{{data.size()},{data.size()/out.size()}}
                    , [=](sycl::nd_item<1> i)
                      {
                        auto lid = i.get_local_id(0);
                        auto rid = i.get_group_linear_id();

                        int x;
                        if(lid == 3) x = 100*(1+rid);

                        int y = sycl::group_broadcast(i.get_group(),x,3);

                        rd[rid] = y;
                      }
                    );
    });
  }

  for(auto e : out)
    std::cout << e << " ";
  std::cout << std::endl;
}
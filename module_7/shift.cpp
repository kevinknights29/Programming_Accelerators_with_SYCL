#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

int main()
{
  std::vector<int>  data = {5,1,2,3,5,9,10,11,5,6,7,1,2,3,9,12};
  std::vector<int>  out(data.size()+2);

  sycl::queue q;
  {
    sycl::buffer values(data);
    sycl::buffer output(out);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor dv{values, h, sycl::read_only};
      sycl::accessor rd{output, h};

      h.parallel_for( sycl::nd_range{{data.size()},{data.size()}}
                    , [=](sycl::nd_item<1> i)
                      {
                        int g = i.get_global_id(0);

                        int x0 = dv[g];
                        int x1 = sycl::shift_group_left(i.get_sub_group(),x0,1);

                        if(g+1<rd.size())
                          rd[g+1] = x1 - x0;
                      }
                    );
    });
  }

  for(auto e : data)
    std::cout << e << " ";
  std::cout << "\n";

  for(auto e : out)
    std::cout << e << " ";
  std::cout << "\n";
}
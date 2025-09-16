#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

int main()
{
  constexpr auto size = 4;
  std::vector<int> data{1,2,3,4,5,6,7,8,9,9,9,9,10,10,10,10};
  std::vector<int> out_all(size), out_any(size), out_none(size);

  sycl::queue q;
  {
    sycl::buffer values(data);
    sycl::buffer output_all(out_all);
    sycl::buffer output_any(out_any);
    sycl::buffer output_none(out_none);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor dv{values, h, sycl::read_only};
      sycl::accessor rall{output_all, h};
      sycl::accessor rany{output_any, h};
      sycl::accessor rnone{output_none, h};

      h.parallel_for( sycl::nd_range{{data.size()},{data.size()/size}}
      , [=](sycl::nd_item<1> i)
        {
          int  x   = dv[i.get_global_id(0)];
          auto g   = i.get_group();
          auto lid = i.get_group_linear_id();

          rall[lid]  = sycl::all_of_group(g,x,[](auto v) { return v < 9;});
          rany[lid]  = sycl::any_of_group(g,x,[](auto v) { return v%2 == 0;});
          rnone[lid] = sycl::none_of_group(g,x,[](auto v) { return v < 8;});
        }
      );
    });
  }

  for(auto e : out_all)
    std::cout << e << " ";
  std::cout << std::endl;

  for(auto e : out_any)
    std::cout << e << " ";
  std::cout << std::endl;

  for(auto e : out_none)
    std::cout << e << " ";
  std::cout << std::endl;
}
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

      auto nb_groups   = sycl::range{out.size()};
      auto group_size  = sycl::range{data.size()/out.size()};
      sycl::local_accessor<int> cache{group_size,h};

      h.parallel_for_work_group( nb_groups, group_size,
        [=](sycl::group<1> grp)
        {
          auto i = grp.get_group_id(0);

          // Implicit barrier (potentially optimized)
          grp.parallel_for_work_item([&](sycl::h_item<1> idx)
          {
            auto gi = idx.get_local_id(0);
            cache[gi] = dv[group_size*i+gi];
          });
          // Implicit barrier

          int sum = 0;
          for(auto e : cache)
            sum += e;

          rd[i] = sum;
        });
    });
  }

  int total = 0;

  for(auto e : out)
    total += e;

  std::cout << total << std::endl;
}
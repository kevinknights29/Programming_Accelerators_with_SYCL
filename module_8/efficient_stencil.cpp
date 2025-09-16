#include <iostream>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>

int main()
{
  constexpr std::size_t size = 16;

  std::vector<int>    data(size);
  std::vector<float>  results(size, 0.f);

  std::random_device  dev;
  std::mt19937        eng{dev()};
  std::uniform_int_distribution dist {1,10};
  std::generate(data.begin(), data.end(), [&]() { return dist(eng); });

  for(auto& e : data)
    std::cout << e << " ";
  std::cout << "\n";

  sycl::queue q;
  {
    sycl::buffer values(data);
    sycl::buffer outs(results);

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor in{values, h, sycl::read_only};
      sycl::accessor out{outs, h};

      sycl::range block{4};
      sycl::range radius{1};
      auto tile = sycl::local_accessor<float,1>{block+2*radius, h};

      h.parallel_for( sycl::nd_range{sycl::range{size},block}
                    , [=](sycl::nd_item<1> i)
                      {
                        auto r   = radius.size();
                        auto lid = i.get_local_id(0);
                        auto gid = i.get_global_id(0);

                        for(std::size_t n=lid;n<tile.size();n++)
                          tile[n] = in[n-r+block.size()*i.get_group(0)];

                        sycl::group_barrier(i.get_group());

                        if(gid >= r && gid < size-r )
                        {
                          float v = 0;
                          for(std::size_t n=0;n<2*r+1;n++)
                            v += tile[lid+n];

                          out[gid] = v/static_cast<float>(2*r+1);
                        }
                      }
                    );
    });
  }


  for(auto e : results)
    std::cout << e << " ";
  std::cout << "\n";
  return 0;
}
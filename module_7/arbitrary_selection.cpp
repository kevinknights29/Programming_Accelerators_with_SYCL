#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

template<typename T> struct matrix
{
  matrix(int n) : storage_(n*n), size_(n) {}

  matrix(int n, std::initializer_list<T> values)
        : storage_{values}, size_(n)
  {}

  T& operator()(int i, int j)       { return storage_[i+j*size_]; }
  T  operator()(int i, int j) const { return storage_[i+j*size_]; }

  decltype(auto) data()       { return storage_.data(); }
  decltype(auto) data() const { return storage_.data(); }

  private:
  std::vector<T>    storage_;
  int               size_;
};


int main()
{
  matrix<int> data( 4, { 1,2,3,4
                        ,5,6,7,8
                        ,9,9,-9,-9
                        ,-10,-10,10,10
                        }
                  );
  matrix<int> out(4);

  sycl::queue q;
  {
    sycl::buffer values(data.data(),sycl::range(4,4));
    sycl::buffer output(out.data(),sycl::range(4,4));

    q.submit([&](sycl::handler& h)
    {
      sycl::accessor dv{values, h, sycl::read_only};
      sycl::accessor rd{output, h};

      h.parallel_for( sycl::nd_range<2>{{4,4},{2,2}}
                  , [=](sycl::nd_item<2> i)
                    {
                      auto gx = i.get_global_id(0);
                      auto gy = i.get_global_id(1);
                      auto lx = i.get_local_id(0);
                      auto ly = i.get_local_id(1);

                      rd[gy][gx] = sycl::select_from_group( i.get_sub_group()
                                                      , dv[gx][gy]
                                                      , ly+2*lx
                                                      );
                    }
                  );
    });
  }

  for(std::size_t j = 0;j < 4;++j)
  {
    for(std::size_t i = 0;i < 4;++i)
        std::cout << data(i,j) << " ";
    std::cout << "\n";
  }

  std::cout << "\n";

  for(std::size_t j = 0;j < 4;++j)
  {
    for(std::size_t i = 0;i < 4;++i)
        std::cout << out(i,j) << " ";
    std::cout << "\n";
  }
}
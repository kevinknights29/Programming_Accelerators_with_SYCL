#include <sycl/sycl.hpp>
#include <initializer_list>
#include <vector>

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
  sycl::queue Q;

  matrix<float> A_h ( 4 , { 19,-21,-23,18
                          , 23, -9,-21, 4
                          ,  2,  6, 14, 6
                          ,  1,  3,-19,16
                          }
                    );

  matrix<float> B_h ( 4 , {-1, 4,3,-1
                          ,-4, 3,2, 3
                          , 1,-1,3,-2
                          , 2,-2,3, 2
                          }
                    );
  matrix<float> C_h(4);

  {
    sycl::buffer A_d(A_h.data(),sycl::range(4,4));
    sycl::buffer B_d(B_h.data(),sycl::range(4,4));
    sycl::buffer C_d(C_h.data(),sycl::range(4,4));

    Q.submit([&](sycl::handler& h)
    {
        sycl::accessor<float,2> A{A_d, h};
        sycl::accessor<float,2> B{B_d, h};
        sycl::accessor<float,2> C{C_d, h};

        auto global = sycl::range{4,4};
        auto local  = sycl::range{2,2};
        h.parallel_for( sycl::nd_range{global,local}
                      , [=](sycl::nd_item<2> idx)
                        {
                          auto i = idx.get_global_id(0);
                          auto j = idx.get_global_id(1);
                          for(int k=0;k<4;k++)
                            C[i][j] += A[i][k] * B[k][j];
                        }
                  );
    });
  }

  for(std::size_t j = 0;j < 4;++j)
  {
    for(std::size_t i = 0;i < 4;++i)
        std::cout << C_h(i,j) << " ";
    std::cout << "\n";
  }
}
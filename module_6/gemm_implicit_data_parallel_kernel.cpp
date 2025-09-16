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

  matrix<float> A_h(3, {2,1,3,4,-1,0,-7,2,1} );
  matrix<float> B_h(3, {-1,5,3,-4,23,12,1,-11,-6});
  matrix<float> C_h(3);

  {
    sycl::buffer A_d(A_h.data(),sycl::range(3,3));
    sycl::buffer B_d(B_h.data(),sycl::range(3,3));
    sycl::buffer C_d(C_h.data(),sycl::range(3,3));

    Q.submit([&](sycl::handler& h)
    {
        sycl::accessor<float,2> A{A_d, h};
        sycl::accessor<float,2> B{B_d, h};
        sycl::accessor<float,2> C{C_d, h};

        h.parallel_for( sycl::range{3,3}
                      , [=](auto idx)
                        {
                          auto i = idx[0];
                          auto j = idx[1];
                          for(int k=0;k<3;k++)
                            C[i][j] += A[i][k] * B[k][j];
                        }
                  );
    });
  }

  for(std::size_t j = 0;j < 3;++j)
  {
    for(std::size_t i = 0;i < 3;++i)
        std::cout << C_h(i,j) << " ";
    std::cout << "\n";
  }
}
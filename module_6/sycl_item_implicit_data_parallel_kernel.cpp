#include <sycl/sycl.hpp>
#include <initializer_list>
#include <vector>

template<typename T> struct matrix
{
  matrix(int n) : storage_(n*n), size_(n) {}
  matrix(int n, std::initializer_list<T> values)
        : storage_{values}, size_(n) {}

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

  constexpr int sz = 10;
  matrix<float> m_h(sz);

  {
    sycl::buffer m_d(m_h.data(),sycl::range(sz,sz));

    Q.submit([&](sycl::handler& h)
    {
        sycl::accessor<float,2> m{m_d, h};

        h.parallel_for( sycl::range{sz,sz}
                      , [=](sycl::item<2> idx)
                        {
                          m[idx] = idx.get_linear_id();
                        }
                  );
    });
  }

  for(std::size_t j = 0;j < sz;++j)
  {
    for(std::size_t i = 0;i < sz;++i)
        std::cout << m_h(i,j) << " ";
    std::cout << "\n";
  }
}
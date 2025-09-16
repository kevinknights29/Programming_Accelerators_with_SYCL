#include <sycl/sycl.hpp>

// Square image made of . and X
struct bitmap
{
  bitmap(std::size_t n) : storage_(n*n,'.'), size_(n) {}
  bitmap(std::size_t n, std::initializer_list<char> values) : storage_{values}, size_(n) {}

  char& operator()(int i, int j)       { return storage_[i+j*size_]; }
  char  operator()(int i, int j) const { return storage_[i+j*size_]; }

  decltype(auto) data()       { return storage_.data(); }
  decltype(auto) data() const { return storage_.data(); }
  
  std::size_t size() const { return size_; }

  private:
  std::vector<char>  storage_;
  std::size_t       size_;
};

void print(char const* name, bitmap const& m)
{
  std::cout << name << ": \n";
  for(std::size_t y=0;y<m.size();y++)
  {
    for(std::size_t x=0;x<m.size();x++)
      std::cout << m(x,y) << " ";
    std::cout << std::endl;
  } 
  std::cout << std::endl;
}

void local_dilation( std::size_t i, std::size_t j
                   , auto acc_in, auto acc_out 
                   , std::size_t sz
                   )
{
  // Dilation counts the number of X in a 3x3 zone centered around (i,j). 
  // If this count is greater than 3, the X is put in the output image.
  auto count = 0;
  std::size_t idx = 0;
  for(int dj=-1;dj<=1;dj++)
  {
    for(int di=-1;di<=1;di++)
    {
      std::size_t ni = i + di;
      std::size_t nj = j + dj;
      if ((ni >= 0 && nj >= 0) && (ni < sz && nj < sz)) {
        idx = ni + nj * sz;
        if (acc_in[idx] == 'X') count++;
      }
    }
  }

  idx = i + j * sz;
  if (count >= 3) {
    acc_out[idx] = 'X';
  } else {
    acc_out[idx] = '.';
  }
  
}

void dilation(sycl::queue& q, bitmap const& X, bitmap& Y, std::size_t sz)
{
  // Constructs two buffers X and Y
  sycl::buffer X_b(X.data(), sycl::range<1>(sz * sz));
  sycl::buffer Y_b(Y.data(), sycl::range<1>(sz * sz));

  q.submit([&](sycl::handler& h)
  {
    // Associate buffers and accessors
    sycl::accessor X_a{X_b, h};
    sycl::accessor Y_a{Y_b, h};

    // Computation over the whole grid is made in group of 3x3 values
    std::size_t grp_sz = 3;
    auto nb_groups   = sycl::range{sz/grp_sz, sz/grp_sz};
    auto group_size  = sycl::range{grp_sz, grp_sz};
    
    // Start a hierarchical kernel to call local_dilation on all points
    h.parallel_for_work_group(nb_groups, [=](sycl::group<2> grp){
        auto ib = grp.get_group_id(0);
        auto jb = grp.get_group_id(1);

        grp.parallel_for_work_item(group_size, [&](sycl::h_item<2> idx){
          auto i = ib * grp_sz + idx.get_local_id(0);
          auto j = jb * grp_sz + idx.get_local_id(1);
          local_dilation(i, j, X_a, Y_a, sz);
        });
      });
  });
}

void exercice(sycl::queue q)
{
  std::size_t sz = 12;
  bitmap X =  { sz, { '.','.','.','.','.','.','.','.','.','.','.','.'
                    , '.','.','X','X','.','.','.','.','X','X','.','.'
                    , '.','.','.','X','.','.','.','.','X','.','.','.'
                    , '.','.','.','X','.','X','X','.','X','.','.','.'
                    , '.','.','.','X','X','X','X','X','X','.','.','.'
                    , '.','.','.','X','X','.','.','X','X','.','.','.'
                    , '.','.','.','X','X','.','.','X','X','.','.','.'
                    , '.','.','.','X','X','X','X','X','X','.','.','.'
                    , '.','.','.','X','.','X','X','.','X','.','.','.'
                    , '.','.','.','X','.','.','.','.','X','.','.','.'
                    , '.','.','X','X','.','.','.','.','X','X','.','.'
                    , '.','.','.','.','.','.','.','.','.','.','.','.'
                    }
            };

  bitmap Y(sz);

  dilation(q,X,Y,sz);

  print("Input", X);
  print("Output", Y);
}

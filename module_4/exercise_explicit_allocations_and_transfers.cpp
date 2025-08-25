#include <sycl/sycl.hpp>

void print(char const* name, auto const& arr)
{
  std::cout << name << ": ";
  for(auto e : arr) std::cout << e << " ";
  std::cout << std::endl;
}

int* alloc_on_device(std::size_t n, sycl::queue& q)
{
  // Allocate n integers on the devide provided by q
  return sycl::malloc_device<int>(n, q);
}

void  transfer(sycl::queue& q, std::vector<int> const& in, int* din)
{
  // Copy data from in to the device pointer din
  q.memcpy(din, in.data(), in.size() * sizeof(int)); 
}

void  transfer( sycl::queue& q, int* dout, std::vector<int>& out)
{
  // Copy data from the device pointer dout in to the data from out
  q.memcpy(out.data(), dout, out.size() * sizeof(int));
}

void process(sycl::queue& q, int* in1, int* in2, int* out1, int* out2, std::size_t n)
{
  // Compute the minimum between elements of in1 and in2
  q.parallel_for( n
                , [in1,in2,out1](auto i) { out1[i] = in1[i] < in2[i] ? in1[i] : in2[i]; }
                );

  // Compute the maximum between elements of in1 and in2
  q.parallel_for( n
                , [in1,in2,out2](auto i) { out2[i] = in1[i] > in2[i] ? in1[i] : in2[i];  }
                );
}


void exercice(sycl::queue& q)
{
  std::size_t size = 10;
  
  std::vector<int> in1(size) , in2(size);
  std::vector<int> out1(size), out2(size);

  for(std::size_t i=0;i<size;++i)
  {
    in1[i] = i%5;
    in2[i] = (i+2)%3;
  }

  // ALlocate 2 device arrays to copy in1 and in2 into
  int* din1  = alloc_on_device(size, q);
  int* din2  = alloc_on_device(size, q);

  // ALlocate 2 device arrays to copy out1 and out2 into
  int* dout1 = alloc_on_device(size, q);
  int* dout2 = alloc_on_device(size, q);

  print("IN1", in1);
  print("IN2", in2);
  
  // Transfer in1/in2 to din1/din2
  transfer(q, in1, din1);
  transfer(q, in2, din2);
  
  q.wait();
  
  process (q,din1 , din2 ,dout1, dout2,size);
  q.wait();
  
  // Transfer dout1/dout2 to out1/out2
  transfer(q, dout1, out1);
  transfer(q, dout2, out2);
  
  q.wait();

  print("OUT1", out1);
  print("OUT2", out2);

  sycl::free(din1,q);
  sycl::free(din2,q);
  sycl::free(dout1,q);
  sycl::free(dout2,q);
}
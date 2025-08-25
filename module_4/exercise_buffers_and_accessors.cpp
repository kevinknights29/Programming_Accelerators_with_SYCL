#include <sycl/sycl.hpp>

void print(char const* name, auto const& arr)
{
  std::cout << name << ": ";
  for(auto e : arr) std::cout << e << " ";
  std::cout << std::endl;
}

void process(sycl::queue& q
            , sycl::buffer<int>&  in1, sycl::buffer<int>&  in2
            , sycl::buffer<int>& out1, sycl::buffer<int>& out2
            )
{
  // Compute the minimum between elements of in1 and in2
  q.submit([&](sycl::handler& h)
    {
      // Declare accessors
      sycl::accessor<int, 1> a_in1{in1, h};
      sycl::accessor<int, 1> a_in2{in2, h};
      sycl::accessor<int, 1> a_out1{out1, h};
      h.parallel_for( out1.get_range()
                    , [=](auto i) { a_out1[i] = a_in1[i] < a_in2[i] ? a_in1[i] : a_in2[i]; }
                    );
    });

  // Compute the maximum between elements of in1 and in2
  q.submit([&](sycl::handler& h)
    {
      // Declare accessors
      sycl::accessor<int, 1> a_in1{in1, h};
      sycl::accessor<int, 1> a_in2{in2, h};
      sycl::accessor<int, 1> a_out2{out2, h};
      h.parallel_for( out2.get_range()
                    , [=](auto i) { a_out2[i] = a_in1[i] > a_in2[i] ? a_in1[i] : a_in2[i]; }
                    );
    });
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

  print("IN1 ", in1);
  print("IN2 ", in2);

  {
    // Initialize buffers to refer to in1, in2, out1 and out2
    sycl::buffer b_in1(in1);
    sycl::buffer b_in2(in2);
    sycl::buffer b_out1(out1);
    sycl::buffer b_out2(out2);

    // Call the process function
    process(q, b_in1, b_in2, b_out1, b_out2);
    
  }

  print("OUT1", out1);
  print("OUT2", out2);
}
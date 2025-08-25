namespace sycl
{
  template< typename  T
          , int       Dimensions = 1
          , typename  AllocatorT = buffer_allocator<std::remove_const_t<T>>
          >
  class buffer;
}

// Usage:
// sycl::buffer<int,1>   b1{ sycl::range<1>{15} };                         // 1

// sycl::buffer<float,3> b2{ sycl::range{5,3,7} };                         // 2

// std::allocator<float> a;
// sycl::buffer<float,2,std::allocator<float>> b3{ sycl::range{5,5}, a };  // 3

// Other use-cases:
// float data[] = {0.f, 0.1f, 0.2f, 0.4f, 0.8f, 1.6f, 3.2f, 6.4f, 12.8f}
// sycl::buffer<float> b4{ data, sycl::range{9} };                       // 1

// std::vector<int> v = {1,10,100,1000,10000};
// sycl::buffer<int> b5{v};                                              // 2

// std::list<int> l = {1,10,100,1000,10000};
// sycl::buffer<int> b5{l.begin(), l.end()};                             // 3

// auto ptr = std::make_shared<double>(42.);
// sycl::buffer<double> b6{ptr };                                        // 4

// Multi-dimensional
// // 1. Explicit version
// std::vector<float> values = {1,2,3,4,5,6,7,8,9,10,11,12};
// sycl::buffer<float,2> b7(values.data(),sycl::range(3,4));

// // 2. Self-driven version
// int data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
// sycl::buffer          b8(data,sycl::range(2,2,3));

// Sub-buffer
// // Initial buffer
// sycl::buffer<float,2> b9(values.data(),sycl::range(6,6));

// sycl::buffer          b9a(b9, sycl::id{0,0}, sycl::range(3,6));
// sycl::buffer          b9b(b9, sycl::id{3,0}, sycl::range(3,6));
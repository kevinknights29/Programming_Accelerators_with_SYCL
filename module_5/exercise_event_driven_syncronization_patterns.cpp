#include <sycl/sycl.hpp>

void print(char const* name, float* arr, int sz)
{
  std::cout << name << ": ";
  for(int i=0;i<sz;++i) std::cout << arr[i] << " ";
  std::cout << std::endl;
}

sycl::event y_pattern(sycl::queue& q, float* X, float* Y, std::size_t sz)
{
  // Fill X and Y asynchronously
  sycl::event ex = q.fill(X, 2.f, sz);
  sycl::event ey = q.fill(Y, 10.f, sz);

  // Compute Y = Y * X once filling is done
  return q.parallel_for( sz, {ex, ey}, [=](auto i) { Y[i] *= X[i]; } );
}

sycl::event yy_pattern(sycl::queue& q, float* X, float* Y, float* Z, std::size_t sz)
{
  // Fill X, Y and Z asynchronously
  sycl::event ex = q.fill(X, 2.f, sz);
  sycl::event ey = q.fill(Y, 10.f, sz);
  sycl::event ez = q.fill(Z, 35.f, sz);

  // Compute Y = Y * X once filling is done
  sycl::event ec = q.parallel_for( sz, {ex, ey}, [=](auto i) { Y[i] *= X[i]; } );
  
  // Compute Z = Z - X once filling is done
  sycl::event ed = q.parallel_for( sz, {ex, ez}, [=](auto i) { Z[i] -= X[i]; } );  

  // Compute X = Y + Z once everything is available
  return q.parallel_for( sz, {ec, ed}, [=](auto i) { X[i] = Y[i] + Z[i]; } );
}

void exercice(sycl::queue& q)
{
  constexpr auto sz = 10;
  // Use shared memory allocation for X, Y and Z
  float* X = sycl::malloc_shared<float>(sz, q);
  float* Y = sycl::malloc_shared<float>(sz, q);
  float* Z = sycl::malloc_shared<float>(sz, q);

  // Test y_pattern
  auto e = y_pattern(q,X,Y,sz);
  // Await event e
  q.wait();

  print("Y Pattern result:", Y, sz);

  // Test yy_pattern
  e = yy_pattern(q,X,Y,Z,sz);
  // Await event e
  q.wait();

  print("YY Pattern result:", X, sz);
}
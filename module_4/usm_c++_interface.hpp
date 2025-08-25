namespace sycl
{
  // Allocates count T values in device memory managed by q
  template<typename T>
  T* malloc_device(size_t count, queue const& q);

  // Allocates count T values in host memory managed by q
  template<typename T>
  T* malloc_host(size_t count, queue const& q);

  // Allocates count T values in shared memory managed by q
  template<typename T>
  T* malloc_shared(size_t count, queue const& q);

  // Allocates count T values in kind memory managed by q
  template<typename T>
  T* malloc(size_t count, queue const& q, usm::alloc kind);

  // Allocates count T values to an alt-aligned address
  // in the device memory managed by q
  template<typename T>
  T* aligned_malloc_device(size_t alt, size_t count, queue const& q);

  // Allocates count T values to an alt-aligned address
  // in the host memory managed by q
  template<typename T>
  T* aligned_malloc_host(size_t alt, size_t count, queue const& q);

  // Allocates count T values to an alt-aligned address
  // in the shared memory managed by q
  template<typename T>
  T* aligned_malloc_shared(size_t alt, size_t count, queue const& q);

  // Allocates count T values to an alt-aligned address
  // in the kind memory managed by q
  template<typename T>
  T* aligned_malloc( size_t alt, size_t count
                   , queue const& q, usm::alloc kind
                   );
}
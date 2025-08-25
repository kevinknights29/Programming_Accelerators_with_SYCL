namespace sycl
{
  // Allocates size bytes in device memory managed by q
  void* malloc_device(size_t size, queue const& q);

  // Allocates size bytes in host memory managed by q
  void* malloc_host(size_t size, queue const& q);

  // Allocates size bytes in shared memory managed by q
  void* malloc_shared(size_t size, queue const& q);

  namespace usm
  {
    // Kind values usable in malloc
    enum class alloc { host, device, shared, unknown };
  }

  // Allocates size bytes in kind memory managed by q
  void* malloc(size_t size, queue const& q, usm::alloc kind);

  // Allocate size bytes to an alt-aligned address
  // in the device memory managed by q
  void* aligned_malloc_device(size_t alt, size_t size, queue const& q);

  // Allocate size bytes to an alt-aligned address
  // in the host memory managed by q
  void* aligned_malloc_host(size_t alt, size_t size, queue const& q);

  // Allocate size bytes to an alt-aligned address
  // in the shared memory managed by q
  void* aligned_malloc_shared(size_t alt, size_t size, queue const& q);

  // Allocate size bytes to an alt-aligned address
  // in the kind memory managed by q.
  void* aligned_malloc( size_t alt, size_t size
                      , queue const& q, usm::alloc kind
                      );

  // Deallocates the memory managed by q and pointed to by ptr
  void* free(void* ptr, queue const& q);
}
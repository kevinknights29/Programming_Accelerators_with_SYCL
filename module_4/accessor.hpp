namespace sycl
{
  template< typename  T
          , int       Dimensions = 1
          , sycl::access_mode AccessMode = /* automatically deduced*/
          , sycl::target AccessTarget    = sycl::target::device,
          >
  class accessor;
}

// Usage:

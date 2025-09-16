// ===== NAIVE =====

sycl::queue Q;

// A, B , C are defined here

Q.submit([&](sycl::handler& h)
{
  h.single_task([=]()
  {
    for(int m=0;m < M; m++)
    {
      for(int n=0;n < N; n++)
      {
        float s = 0;

        for(int k=0;k < K; k++)
          s += A[m*k+k] * B[n+ N*k];

        C[m*N+n] = s;
      }
    }
  });
}

// ===== PARALLEL_FOR 1D =====

sycl::queue Q;

// A, B , C are defined here

Q.submit([&](sycl::handler& h)
{
  h.parallel_for(sycl::range{M}, [=]( sycl::id<1> index)
  {
    int m = index[0];

    for(int n=0;n < N; n++)
    {
      float s = 0;

      for(int k=0;k < K; k++)
        s += A[m*k+k] * B[n+ N*k];

      C[m*N+n] = s;
    }
  });
}

// ===== PARALLEL_FOR 2D =====

sycl::queue Q;

// A, B , C are defined here
Q.submit([&](sycl::handler& h)
{
  h.parallel_for(sycl::range{M,N}, [=](sycl::id<2> index)
  {
    int m = index[0];
    int n = index[1];
    float s = 0;

    for(int k=0;k < K; k++)
      s += A[m*k+k] * B[n+ N*k];

    C[m*N+n] = s;
  });
}

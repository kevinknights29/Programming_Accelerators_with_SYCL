#include <sycl/sycl.hpp>
#include <vector>

int main() 
{
    sycl::queue Q;
    std::vector<float> D(15);

    {
    sycl::buffer B(S);

    Q.submit([&](sycl::handler& h)
    {
        sycl::accessor   x{B, h};                                     //  1.
        sycl::accessor xrw{B, h, sycl::read_write};                   //  2.
        sycl::accessor  xr{B, h, sycl::read_only };                   //  3.
        sycl::accessor  xw{B, h, sycl::write_only};                   //  4.
        sycl::accessor  xw0{B, h, sycl::read_write, sycl::no_init};   //  5.
    });
    }
}

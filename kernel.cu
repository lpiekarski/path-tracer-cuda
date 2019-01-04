/*#define RTC_TIMER
#define RTC_PRECISION_DOUBLE
#include "tracer.h"
#include "tclist.h"
#include "color.h"
#include "tracable.h"
#include "triangle.h"
#include "matrix.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <memory>
#include <string>

using namespace rtc;

const size_t bpp = 3;

tracer<tracable_list<3, bpp>, 3, bpp> trc;

int main() {
    //rtc::vec_type off = 0;
    vec3 v1 = vec3(3, 1, 1),
        v2 = vec3(2, -1, 1),
        v3 = vec3(3, -1, -1),
        v4 = vec3(2, 1, -1);
    triangle<3, bpp> t1 = triangle<3, bpp>(
        std::make_shared<vec3>(v1),
        std::make_shared<vec3>(v2),
        std::make_shared<vec3>(v3),
        color<bpp>(1, 1, 1), color<bpp>(1, 1, 1));
    triangle<3, bpp> t2 = triangle<3, bpp>(
        std::make_shared<vec3>(v1),
        std::make_shared<vec3>(v4),
        std::make_shared<vec3>(v3),
        color<bpp>(1, 1, 1), color<bpp>(1, 1, 1));
    trc.add_tracable(t1);
    trc.add_tracable(t2);

    trc.draw_bitmap<4>(1080, 1080)[0].write("test.bmp");
    //system("test.bmp");
    //vec3 v = (vec3(1, 2, 3) + 4) * 5 * vec3(2, 2, 0.5);
}*/
#define RTC_USE_CUDA
#include <cuda_runtime.h>
#include "defs.h"
#include <array>

const int threads_num = 20;

__global__ void test() {
    __shared__ float a[threads_num];//rtc::_array<float, threads_num> a;
    a[threadIdx.x] = 0;
    __syncthreads();
    for (size_t i = 0; i < threads_num; ++i)
        ++a[i];
    __syncthreads();
    for (size_t i = 0; i < threads_num; ++i)
        printf("%d %f\n", threadIdx.x, a[threadIdx.x]);
    
}

int main() {
    //auto d_arr = rtc::_array<int, 5>::device_ctr();
    
    test<<<1, threads_num>>>();
    cudaDeviceSynchronize();
}
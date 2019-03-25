#define RTC_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "tclist.h"
#include "tracer.h"

using namespace rtc;
using namespace std;

int main() {
    rtc::tracer<rtc::tracable_list<3, 3>, 3, 3> t;
    rtc::tracer<rtc::tracable_list<3, 3>, 3, 3> *d_t = rtc::tracer<rtc::tracable_list<3, 3>, 3, 3>::device_ctr(t);
    auto bmps = d_t->device_draw_bitmap(800, 600);
    bmps[0].write("gpu_test.bmp");

    rtc::tracer<rtc::tracable_list<3, 3>, 3, 3>::device_dtr(d_t);
}
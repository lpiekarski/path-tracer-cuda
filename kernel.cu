#define RTC_TIMER
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

/*
void add_cube(vec_type offset) {
vec3 v1 = vec3(offset - 1, offset - 1, offset - 1),
v2 = vec3(offset - 1, offset - 1, offset + 1),
v3 = vec3(offset - 1, offset + 1, offset - 1),
v4 = vec3(offset - 1, offset + 1, offset + 1),
v5 = vec3(offset + 1, offset - 1, offset - 1),
v6 = vec3(offset + 1, offset - 1, offset + 1),
v7 = vec3(offset + 1, offset + 1, offset - 1),
v8 = vec3(offset + 1, offset + 1, offset + 1),
Triangle t1(),
t2(),
t3(),
t4(),
t5(),
t6(),
t7(),
t8()
}*/

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
}
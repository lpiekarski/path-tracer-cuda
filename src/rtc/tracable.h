#ifndef __TRACABLE_H__
#define __TRACABLE_H__

#include "rtc.h"
#include "color.h"
#include "vector.h"
#include "ray.h"

_RTC_BEGIN
template <size_t _Dims,
    size_t BPP>
    class tracable {
    public:
        _DEVHOST virtual bool ray_intersection(const ray<_Dims>& r,
            vec_type& intersection_dist) = 0;

        _DEVHOST virtual color<BPP> ambient_color(const ray<_Dims>& r,
            const vector<_Dims>& intersection_point) = 0;

        _DEVHOST virtual color<BPP> diffuse_color(const ray<_Dims>& r,
            const vector<_Dims>& intersection_point) = 0;

        _DEVHOST virtual ray<_Dims> get_reflection(const ray<_Dims>& r,
            const vector<_Dims>& intersection_point) = 0;
};
_RTC_END

#endif /* __TRACABLE_H__ */
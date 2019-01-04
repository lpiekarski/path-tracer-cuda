#ifndef __TRACABLE_H__
#define __TRACABLE_H__

#include "rtc.h"

_RTC_BEGIN
template <size_t N,
    size_t BPP>
    class tracable {
    public:
        virtual bool ray_intersection(const ray<N>& r,
            vec_type& intersection_dist) = 0;

        virtual color<BPP> ambient_color(const ray<N>& r,
            const vector<N>& intersection_point) = 0;

        virtual color<BPP> diffuse_color(const ray<N>& r,
            const vector<N>& intersection_point) = 0;

        virtual ray<N> get_reflection(const ray<N>& r,
            const vector<N>& intersection_point) = 0;
};
_RTC_END

#endif /* __TRACABLE_H__ */
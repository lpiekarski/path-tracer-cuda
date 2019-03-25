#ifndef __TCLIST_H__
#define __TCLIST_H__

#include "rtc.h"
#include "tracable.h"
#include "../cuda_ext/type_traits.h"

_RTC_BEGIN
    // CLASS TEMPLATE tracable_list
template <size_t _Dims,
    size_t BPP>
    class tracable_list {
    private:
        _vector<tracable<_Dims, BPP> *> tracables;

    public:
        _DEVHOST tracable_list() : tracables() {}

        _DEVHOST tracable_list(const tracable_list& other)
            : tracables(other.tracables) {}

        _DEVHOST tracable_list& operator=(const tracable_list& other) {
            tracables = other.tracables;
            return *this;
        }

        _DEVHOST void add(tracable<_Dims, BPP>& t) { tracables.push_back(&t); }

        _DEVHOST bool ray_intersection(const ray<_Dims>& r,
            tracable<_Dims, BPP>** tr_obj,
            vec_type& intersection_dist) const {
            intersection_dist = MAX_RAY_LENGTH;
            vec_type dst;
            bool intersection = false;
            for (size_t i = 0; i < tracables.size(); ++i) {
                tracable<_Dims, BPP>* t = tracables[i];
                if (t->ray_intersection(r, dst) && dst < intersection_dist) {
                    *tr_obj = t;
                    intersection_dist = dst;
                    intersection = true;
                }
            }
            return intersection;
        }
    };

template <size_t _Dims,
    size_t BPP>
    struct is_tracable_container<tracable_list<_Dims, BPP>>
    : true_type {};

_RTC_END

#endif /* __TCLIST_H__ */

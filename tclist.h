#ifndef __TCLIST_H__
#define __TCLIST_H__

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "rtc.h"
#include "tracable.h"

_RTC_BEGIN
    // CLASS TEMPLATE tracable_list
template <size_t N,
    size_t BPP>
    class tracable_list {
    private:
        _vector<tracable<N, BPP>*> tracables;

    public:
        tracable_list() : tracables() {}

        tracable_list(const tracable_list& other)
            : tracables(other.tracables) {}

        tracable_list& operator=(const tracable_list& other) {
            tracables = other.tracables;
        }

        void add(tracable<N, BPP>& t) { tracables.push_back(&t); }

        bool ray_intersection(const ray<N>& r,
            tracable<N, BPP>** tr_obj,
            vec_type& intersection_dist) const {
            intersection_dist = MAX_RAY_LENGTH;
            vec_type dst;
            bool intersection = false;
            for (size_t i = 0; i < tracables.size(); ++i) {
                tracable<N, BPP>* t = tracables[i];
                if (t->ray_intersection(r, dst) && dst < intersection_dist) {
                    *tr_obj = t;
                    intersection_dist = dst;
                    intersection = true;
                }
            }
            return intersection;
        }
    };

template <size_t N,
    size_t BPP>
    struct is_tracable_container<tracable_list<N, BPP>> 
    : true_type {};
_RTC_END

#endif /* __TCLIST_H__ */

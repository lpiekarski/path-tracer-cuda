#ifndef __RAY_H__
#define __RAY_H__

#include "rtc.h"
#include "vector.h"

_RTC_BEGIN
    // CLASS TEMPLATE ray
template <size_t N>
    class ray {
    private:
        normal<N> direction;
        vector<N> origin;

    public:
        ray() : direction(), origin() {}

        ray(const vector<N>& direction, const vector<N>& origin)
            : direction(direction),
            origin(origin) {}

        ray(const vector<N>& direction)
            : direction(direction),
            origin() {}

        vector<N> get_point(const vec_type& distance) const {
            return origin + (distance * direction);
        }

        vector<N>& dir() { return direction; }

        vector<N> dir() const { return direction; }

        vector<N>& o() { return origin; }

        vector<N> o() const { return origin; }
    };

    using ray3 = ray<3>;
_RTC_END

#endif /* __RAY_H__ */
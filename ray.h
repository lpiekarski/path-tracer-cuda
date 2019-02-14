#ifndef __RAY_H__
#define __RAY_H__

#include "rtc.h"
#include "vector.h"

_RTC_BEGIN
    // CLASS TEMPLATE ray
template <size_t _Dims>
    class ray {
    private:
        normal<_Dims> direction;
        vector<_Dims> origin;

    public:
        using _Mytype = ray<_Dims>;

        _DEVHOST ray() : direction(), origin() {}

        _DEVHOST ray(const vector<_Dims>& direction, const vector<_Dims>& origin)
            : direction(direction),
            origin(origin) {}

        _DEVHOST ray(const vector<_Dims>& direction)
            : direction(direction),
            origin() {}

        _DEVHOST vector<_Dims> get_point(const vec_type& distance) const {
            return origin + (distance * direction);
        }

        _DEVHOST vector<_Dims>& dir() { return direction; }

        _DEVHOST vector<_Dims> dir() const { return direction; }

        _DEVHOST vector<_Dims>& o() { return origin; }

        _DEVHOST vector<_Dims> o() const { return origin; }
#ifdef RTC_USE_CUDA
        template <typename... _Args,
        typename = enable_if_t<is_constructible<_Mytype, _Args...>::value>>
        _HOST static _Mytype* device_ctr(_Args... args) {
            _Mytype h_ret(args...);
            _Mytype *d_ret_ptr;
            size_t sz = sizeof(_Mytype);
            cudaMalloc((void**)&d_ret_ptr, sz);
            cudaMemcpy(d_ret_ptr, &h_ret, sz, cudaMemcpyHostToDevice);
            return d_ret_ptr;
        }

        _HOST static void device_dtr(_Mytype *ptr) {
            cudaFree(ptr);
        }
#endif /* RTC_USE_CUDA */
    };

    using ray3 = ray<3>;
_RTC_END

#endif /* __RAY_H__ */
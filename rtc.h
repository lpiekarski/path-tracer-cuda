#ifndef __RTC_H__
#define __RTC_H__

#include <cassert>
#include <cstdint>
#include <vector>

#include "type_traits.h"
#include "defs.h"

_RTC_BEGIN
#if defined(RTC_PRECISION_LONG_DOUBLE)
using vec_type = long double;
constexpr vec_type EPSILON = LDBL_EPSILON * 30;
constexpr vec_type SURFACE_EPSILON = 30 * EPSILON;
constexpr vec_type MAX_RAY_LENGTH = LDBL_MAX;

#elif defined(RTC_PRECISION_DOUBLE)
using vec_type = double;
constexpr vec_type EPSILON = DBL_EPSILON;
constexpr vec_type SURFACE_EPSILON = 30 * EPSILON;
constexpr vec_type MAX_RAY_LENGTH = DBL_MAX;

#else
using vec_type = float;
constexpr vec_type EPSILON = FLT_EPSILON * 30;
constexpr vec_type SURFACE_EPSILON = 30 * EPSILON;
constexpr vec_type MAX_RAY_LENGTH = FLT_MAX;

#endif
    // TEMPLATE CLASS is_tracable_container
template <class _Ty>
    struct is_tracable_container : false_type {};

    // FUNCTION TEMPLATE little_endian_insert
template <typename _Ty,
    typename = enable_if_t<is_integral<_Ty>::value>>
    _DEVHOST void little_endian_insert(_Ty t, size_t size, _vector<char> &bytes, size_t offset) {
        const unsigned int byte_mask = 255;
        while (size--) {
            bytes[offset++] = static_cast<char>(t & byte_mask);
            t = t >> 8;
        }
    }

    // FUNCTION TEMPLATE little_endian_read
template <typename _Ty,
    typename = enable_if_t<is_integral<_Ty>::value>>
    _DEVHOST _Ty little_endian_read(size_t size, _vector<char> &bytes, size_t offset) {
        _Ty ret = 0;
        while (size--)
            ret = (ret << 8) + bytes[offset + size];
        return ret;
    }
_RTC_END

#endif /* __RTC_H__ */
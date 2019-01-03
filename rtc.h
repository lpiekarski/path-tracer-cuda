#ifndef __RTC_H__
#define __RTC_H__

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>

#ifdef __cplusplus
#define _RTC_BEGIN namespace rtc {
#define _RTC_END }
#endif

#ifdef RTC_USE_CUDA
#define _DEVICE __device__
#define _HOST __host__
#define _INTER _HOST _DEVICE
#define _VECTOR(T) // TODO
#define _ARRAY(T, N) // TODO

#else /* !RTC_USE_CUDA */
#define _DEVICE
#define _HOST
#define _INTER
#define _VECTOR(T) std::vector<T>
#define _ARRAY(T, N) std::array<T, N>
#endif

_RTC_BEGIN
#if defined(RTC_PRECISION_LONG_DOUBLE)
typedef long double vec_type;
constexpr vec_type EPSILON = LDBL_EPSILON * 30;
constexpr vec_type SURFACE_EPSILON = 30 * EPSILON;
constexpr vec_type MAX_RAY_LENGTH = LDBL_MAX;

#elif defined(RTC_PRECISION_DOUBLE)
typedef double vec_type;
constexpr vec_type EPSILON = DBL_EPSILON;
constexpr vec_type SURFACE_EPSILON = 30 * EPSILON;
constexpr vec_type MAX_RAY_LENGTH = DBL_MAX;

#else
typedef float vec_type;
constexpr vec_type EPSILON = FLT_EPSILON * 30;
constexpr vec_type SURFACE_EPSILON = 30 * EPSILON;
constexpr vec_type MAX_RAY_LENGTH = FLT_MAX;

#endif
    // STRUCT TEMPLATE for_all
template <template <class> class _Test,
    class... _Lty>
    struct for_all;

template <template <class> class _Test>
    struct for_all<_Test> : std::true_type {};

template <template <class> class _Test,
    class _Hty,
    class... _Tty>
    struct for_all<_Test, _Hty, _Tty...> 
        : std::bool_constant<for_all<_Test, _Tty...>::value &&
        _Test<_Hty>::value> {};

    // STRUCT TEMPLATE are_same
template <class _Ty,
    class... _Lty>
    struct are_same;

template <class _Ty>
    struct are_same<_Ty> : std::true_type {};

template <class _Ty, 
    class _Hty, 
    class... _Tty>
    struct are_same<_Ty, _Hty, _Tty...> 
        : std::bool_constant<are_same<_Ty, _Tty...>::value && 
        std::is_same<_Ty, _Hty>::value> {};

    // STRUCT TEMPLATE are_convertible
template <class _To, class... _Lty>
    struct are_convertible;

template <class _To>
    struct are_convertible<_To> : std::true_type {};

template <class _To, 
    class _Hty, 
    class... _Tty>
    struct are_convertible<_To, _Hty, _Tty...>
        : std::bool_constant<are_convertible<_To, _Tty...>::value && 
        std::is_convertible<_Hty, _To>::value> {};

    // ALIAS TEMPLATE are_arithmetic
template <class... _Lty>
    using are_arithmetic = for_all<std::is_arithmetic, _Lty...>;

    // STRUCT TEMPLATE is_tracable_container
template <class _Ty>
    struct is_tracable_container : std::false_type {};

#if _HAS_CXX17
template <template <class> class _Test,
    class... _Lty>
    inline constexpr bool for_all_v = for_all<_Test, _Lty...>::value;

template <class _Ty,
    class... _Lty>
    inline constexpr bool are_same_v = are_same<_Ty, _Lty...>::value;

template <class _To,
    class... _Lty>
    inline constexpr bool are_convertible_v = are_convertible<_To, _Lty...>::value;

template <class... _Lty>
    inline constexpr bool are_arithmetic_v = for_all_v<std::is_arithmetic, _Lty...>;

template <class _Ty>
    inline constexpr bool is_tracable_container_v = is_tracable_container<_Ty>::value;
#endif /* _HAS_CXX17 */

    // FUNCTION TEMPLATE little_endian_insert
template <typename _Ty,
    typename = std::enable_if_t<std::is_integral<_Ty>::value>>
    void little_endian_insert(_Ty t, size_t size, std::vector<char> &bytes, size_t offset) {
        assert(bytes.size() >= size + offset);
        const uint16_t byte_mask = 255;
        while (size--) {
            bytes[offset++] = static_cast<char>(t & byte_mask);
            t = t >> 8;
        }
    }

    // FUNCTION TEMPLATE little_endian_read
template <typename _Ty,
    typename = std::enable_if_t<std::is_integral<_Ty>::value>>
    _Ty little_endian_read(size_t size, std::vector<char> &bytes, size_t offset) {
        assert(bytes.size() >= size + offset);
        _Ty ret = 0;
        while (size--)
            ret = (ret << 8) + bytes[offset + size];
        return ret;
    }
_RTC_END

#endif /* __RTC_H__ */
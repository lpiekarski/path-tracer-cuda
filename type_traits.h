#ifndef __TYPE_TRAITS_H__
#define __TYPE_TRAITS_H__

#ifdef __cplusplus
#define _RTC_BEGIN namespace rtc {
#define _RTC_END }
#else
#define _RTC_BEGIN
#define _RTC_END
#endif

#ifdef RTC_USE_CUDA
#include <cuda_runtime.h>
#define _DEVICE __device__
#define _HOST __host__
#define _DEVHOST _DEVICE _HOST
#else
#define _DEVICE
#define _HOST
#define _DEVHOST
#endif

_RTC_BEGIN
    // TEMPLATE CLASS integral_constant
template <class _Ty,
    _Ty _Val>
    struct integral_constant {
        static constexpr _Ty value = _Val;

        using value_type = _Ty;
        using type = integral_constant<_Ty, _Val>;

        _DEVHOST constexpr operator value_type() const _NOEXCEPT {
            return (value);
        }

        _DEVHOST constexpr value_type operator()() const _NOEXCEPT {
            return (value);
        }
    };

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

    // ALIAS TEMPLATE bool_constant
template <bool _Val>
    using bool_constant = integral_constant<bool, _Val>;

    // TEMPLATE CLASS _Cat_base
template <bool _Val>
    struct _Cat_base
        : integral_constant<bool, _Val> {};

    // TEMPLATE CLASS enable_if
template <bool _Test,
    class _Ty = void>
    struct enable_if {};

template <class _Ty>
    struct enable_if<true, _Ty> {
        using type = _Ty;
    };

template <bool _Test,
    class _Ty = void>
    using enable_if_t = typename enable_if<_Test, _Ty>::type;

    // TEMPLATE CLASS conditional
template <bool _Test,
    class _Ty1,
    class _Ty2>
    struct conditional {
        using type = _Ty2;
    };

template <class _Ty1,
    class _Ty2>
    struct conditional<true, _Ty1, _Ty2> {
        using type = _Ty1;
    };

    // TEMPLATE CLASS is_same
template <class _Ty1,
    class _Ty2>
    struct is_same
        : false_type {};

template <class _Ty1>
    struct is_same<_Ty1, _Ty1>
        : true_type {};

    // TEMPLATE CLASS remove_const
template <class _Ty>
    struct remove_const {
        using type = _Ty;
    };

template <class _Ty>
    struct remove_const<const _Ty> {
        using type = _Ty;
    };

    // TEMPLATE CLASS remove_volatile
template <class _Ty>
    struct remove_volatile {
        using type = _Ty;
    };

template <class _Ty>
    struct remove_volatile<volatile _Ty> {
        using type = _Ty;
    };

    // TEMPLATE CLASS remove_cv
template <class _Ty>
    struct remove_cv {
        using type = remove_const<typename remove_volatile<_Ty>::type>::type;
    };

    // TEMPLATE CLASS _Is_integral
template <class _Ty>
    struct _Is_integral
        : false_type {};

template <>
    struct _Is_integral<bool>
        : true_type {};

template <>
    struct _Is_integral<char>
        : true_type {};

template <>
    struct _Is_integral<unsigned char>
        : true_type {};

template <>
    struct _Is_integral<signed char>
        : true_type {};

#ifdef _NATIVE_WCHAR_T_DEFINED
template <>
    struct _Is_integral<wchar_t>
        : true_type {};
#endif /* _NATIVE_WCHAR_T_DEFINED */

template <>
    struct _Is_integral<unsigned short>
        : true_type {};

template <>
    struct _Is_integral<signed short>
        : true_type {};

template <>
    struct _Is_integral<unsigned int>
        : true_type {};

template <>
    struct _Is_integral<signed int>
        : true_type {};

template <>
    struct _Is_integral<unsigned long>
        : true_type {};

template <>
    struct _Is_integral<signed long>
        : true_type {};

template <>
    struct _Is_integral<char16_t>
        : true_type {};

template <>
    struct _Is_integral<char32_t>
        : true_type {};

template <>
    struct _Is_integral<long long>
        : true_type {};

template <>
    struct _Is_integral<unsigned long long>
        : true_type {};

    // TEMPLATE CLASS is_integral
template <class _Ty>
    struct is_integral
        : _Is_integral<typename remove_cv<_Ty>::type> {};

    // TEMPLATE CLASS _Is_floating_point
template <class _Ty>
    struct _Is_floating_point
        : false_type {};

template <>
    struct _Is_floating_point<float>
        : true_type {};

template <>
    struct _Is_floating_point<double>
        : true_type {};

template <>
    struct _Is_floating_point<long double>
        : true_type {};

    // TEMPLATE CLASS is_floating_point
template <class _Ty>
    struct is_floating_point
        : _Is_floating_point<typename remove_cv<_Ty>::type> {};

    // TEMPLATE CLASS is_arithmetic
template <class _Ty>
    struct is_arithmetic
        : _Cat_base<is_integral<_Ty>::value
        || is_floating_point<_Ty>::value> {};

template<class _From,
        class _To>
        struct is_convertible
        : _Cat_base<__is_convertible_to(_From, _To)> {};

    // TEMPLATE CLASS for_all
template <template <class> class _Test,
    class... _Lty>
    struct for_all;

template <template <class> class _Test>
    struct for_all<_Test> : true_type {};

template <template <class> class _Test,
    class _Hty,
    class... _Tty>
    struct for_all<_Test, _Hty, _Tty...> 
        : bool_constant<for_all<_Test, _Tty...>::value &&
        _Test<_Hty>::value> {};

    // TEMPLATE CLASS are_same
template <class _Ty,
    class... _Lty>
    struct are_same;

template <class _Ty>
    struct are_same<_Ty> : true_type {};

template <class _Ty, 
    class _Hty, 
    class... _Tty>
    struct are_same<_Ty, _Hty, _Tty...> 
        : bool_constant<are_same<_Ty, _Tty...>::value && 
        is_same<_Ty, _Hty>::value> {};

    // TEMPLATE CLASS are_convertible
template <class _To, class... _Lty>
    struct are_convertible;

template <class _To>
    struct are_convertible<_To> : true_type {};

template <class _Ty, class... _Args>
    struct is_constructible
        : _Cat_base<__is_constructible(_Ty, _Args...)> {};

template <class _To, 
    class _Hty, 
    class... _Tty>
    struct are_convertible<_To, _Hty, _Tty...>
        : bool_constant<are_convertible<_To, _Tty...>::value && 
        is_convertible<_Hty, _To>::value> {};

    // ALIAS TEMPLATE are_arithmetic
template <class... _Lty>
    using are_arithmetic = for_all<is_arithmetic, _Lty...>;

#if _HAS_CXX17
template <class _Ty,
    class _Uty>
    inline constexpr bool is_same_v = is_same<_Ty, _Uty>::value;

template <class _Ty>
    inline constexpr bool is_integral_v = is_integral<_Ty>::value;

template <class _Ty>
    inline constexpr bool is_floating_point_v = is_floating_point<_Ty>::value;

template <class _Ty>
    inline constexpr bool is_arithmetic_v = is_arithmetic<_Ty>::value;

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
    inline constexpr bool are_arithmetic_v = for_all_v<is_arithmetic, _Lty...>;

template <class _Ty>
    inline constexpr bool is_tracable_container_v = is_tracable_container<_Ty>::value;
#endif /* _HAS_CXX17 */
_RTC_END

#endif /* __TYPE_TRAITS_H__ */
#ifndef __DH_ARRAY_H__
#define __DH_ARRAY_H__

#include "type_traits.h"

_RTC_BEGIN
template <typename _Ty, size_t _Size>
    class dh_array {
    private:
        using value_type = _Ty;
        using _Mytype = dh_array<_Ty, _Size>;

        _Ty arr[_Size];

    public:
        _DEVHOST dh_array() : arr() {}

        _DEVHOST dh_array(const dh_array& other) {
            for (size_t i = 0; i < _Size; ++i)
                arr[i] = other.arr[i];
        }

        _DEVHOST dh_array(_Ty* _Array) {
            for (size_t i = 0; i < _Size; ++i)
                arr[i] = _Array[i];
        }

    template <typename... _Args, 
        typename = enable_if_t<sizeof...(_Args) == _Size>>
        _DEVHOST dh_array(_Args... args)
            : arr{ static_cast<_Ty>(args)... } {}

        _DEVHOST _Ty operator[](size_t idx) const {
            return arr[idx];
        }

        _DEVHOST _Ty& operator[](size_t idx) {
            return arr[idx];
        }

        _DEVHOST _Ty* data() {
            return arr;
        }

        _DEVHOST const _Ty* data() const {
            return arr;
        }

        _DEVHOST constexpr size_t size() const noexcept {
            return _Size;
        }

#ifdef RTC_USE_CUDA
    template <typename... _Args>
    //typename = enable_if_t<is_constructible<dh_array<_Ty, _Size>, _Args...>
        _HOST static dh_array<_Ty, _Size>* device_ctr(_Args... args) {
            _Mytype h_ret(args...);
            _Mytype *d_ret_ptr;
            size_t sz = sizeof(_Mytype);
            cudaMalloc((void**)&d_ret_ptr, sz);
            cudaMemcpy(d_ret_ptr, &h_ret, sz, cudaMemcpyHostToDevice);
            return d_ret_ptr;
        }
#endif /* RTC_USE_CUDA */
    };
_RTC_END

#endif /* __DH_ARRAY_H__ */
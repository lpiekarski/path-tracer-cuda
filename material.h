#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <cmath>

#include "rtc.h"
#include "bitmap.h"
//TODO: replace shared pointers with c style pointers
_RTC_BEGIN
    // CLASS TEMPLATE material
template <size_t BPP>
    class material {
        using _Mytype = material<BPP>;
    private:
        bitmap<BPP> *diffuse,
            *ambient,
            *reflection,
            *refraction, 
            *normal;
            //subsurface

        _DEVHOST color<BPP> get_from_bmp(const vec_type& x,
            const vec_type& y,
            const color<BPP>& def,
            bitmap<BPP>* bmp) {
            if (bmp == nullptr)
                return def;
            x = x - floor(x);
            y = y - floor(y);
            size_t bmp_x = static_cast<size_t>(x * bmp->get_width());
            size_t bmp_y = static_cast<size_t>(y * bmp->get_height());
            return bmp->get(bmp_x, bmp_y);
        }

    public:
        _DEVHOST material() 
            : diffuse(nullptr),
            ambient(nullptr),
            reflection(nullptr),
            refraction(nullptr),
            normal(nullptr) {}

        _DEVHOST material& set_diffuse(bitmap<BPP> *_Diffuse_bmp) {
            diffuse = _Diffuse_bmp;
            return *this;
        }

        _DEVHOST material& set_ambient(bitmap<BPP> *_Ambient_bmp) {
            ambient = _Ambient_bmp;
            return *this;
        }

        _DEVHOST material& set_reflection(bitmap<BPP> *_Reflection_bmp) {
            reflection = _Reflection_bmp;
            return *this;
        }

        _DEVHOST material& set_refraction(bitmap<BPP> *_Refraction_bmp) {
            refraction = _Refraction_bmp;
            return *this;
        }

        _DEVHOST material& set_normal(bitmap<BPP> *_Normal_bmp) {
            normal = _Normal_bmp;
            return *this;
        }

        _DEVHOST color<BPP> get_diffuse(const vec_type& x,
            const vec_type& y,
            const color<BPP>& def) {
            return get_from_bmp(x, y, def, diffuse);
        }

        _DEVHOST color<BPP> get_ambient(const vec_type& x,
            const vec_type& y,
            const color<BPP>& def) {
            return get_from_bmp(x, y, def, ambient);
        }

        _DEVHOST color<BPP> get_reflection(const vec_type& x,
            const vec_type& y,
            const color<BPP>& def) {
            return get_from_bmp(x, y, def, reflection);
        }

        _DEVHOST color<BPP> get_refraction(const vec_type& x,
            const vec_type& y,
            const color<BPP>& def) {
            return get_from_bmp(x, y, def, refraction);
        }

        _DEVHOST color<BPP> get_normal(const vec_type& x,
            const vec_type& y,
            const color<BPP>& def) {
            return get_from_bmp(x, y, def, normal);
        }
#ifdef RTC_USE_CUDA
        template <typename... _Args>
        typename = enable_if_t<is_constructible<_Mytype, _Args...>::value>
            _HOST static _Mytype* device_ctr(_Args... args) {
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

#endif /* __MATERIAL_H__ */
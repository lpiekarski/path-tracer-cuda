#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "ray.h"
#include "rtc.h"
#include "vector.h"

#define DEFAULT_SCREEN_DIST 2
#define DEFAULT_SCREEN_DIM 1

_RTC_BEGIN
    // CLASS TEMPLATE camera
template <size_t _Dims>
    class camera {
    private:
        _array<normal<_Dims>, _Dims> base;
        vector<_Dims - 1> screen_dims;
        vec_type screen_dist;

    public:
        using _Mytype = camera<_Dims>;

        _DEVHOST camera()
            : base(standard_base<_Dims>()),
            screen_dims(from_value<_Dims - 1>(DEFAULT_SCREEN_DIM)),
            screen_dist(DEFAULT_SCREEN_DIST) {}

        _DEVHOST camera(const vector<_Dims - 1>& screen_dims,
            const vec_type& screen_dist) 
            : base(standard_base<_Dims>()),
            screen_dims(screen_dims),
            screen_dist(screen_dist) {}

    template <typename... _Args,
        typename = enable_if_t<sizeof...(_Args) == _Dims && are_same<vector<_Dims>, _Args...>::value>>
        _DEVHOST camera(const vector<_Dims - 1>& screen_dims,
            const vec_type& screen_dist,
            _Args... base_vecs)
            : screen_dims(screen_dims), 
            screen_dist(screen_dist) {
            base = {base_vecs...};
        }

        _DEVHOST camera(const vector<_Dims - 1>& screen_dims,
            const vec_type& screen_dist,
            vector<_Dims>* arr)
            : base(arr),
            screen_dims(screen_dims),
            screen_dist(screen_dist) {}

        _DEVHOST ray<_Dims> get_ray(vec_type* screen_pos_arr) const {
            vector<_Dims> screen_point;
            screen_point[0] = screen_dist;
            for (size_t i = 1; i < _Dims; ++i)
                screen_point[i] = screen_pos_arr[i - 1] * screen_dims[i - 1];

            vector<_Dims> scr_pt_cam_pov;
            for (size_t i = 0; i < _Dims; ++i)
                scr_pt_cam_pov = scr_pt_cam_pov + (base[i] * screen_point[i]);

            return ray<_Dims>(scr_pt_cam_pov, scr_pt_cam_pov - EPSILON);
        }

    template <typename... _Args, 
        typename = enable_if_t<sizeof...(_Args) == _Dims - 1 && are_same<vec_type, _Args...>::value>>
        _DEVHOST ray<_Dims> get_ray(const _Args&... screen_pos) const {
            _array<vec_type, _Dims - 1> screen_pos_arr;
            screen_pos_arr = { static_cast<vec_type>(screen_pos)... };
            
            return get_ray(screen_pos_arr.data());
        }
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
_RTC_END

#endif /* __CAMERA_H__ */
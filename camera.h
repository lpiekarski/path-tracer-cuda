#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <array>
#include <type_traits>

#include "ray.h"
#include "rtc.h"
#include "vector.h"

#define DEFAULT_SCREEN_DIST 2
#define DEFAULT_SCREEN_DIM 1

_RTC_BEGIN
    // CLASS TEMPLATE camera
template <size_t N>
    class camera {
    private:
        _array<normal<N>, N> base;
        vector<N - 1> screen_dims;
        vec_type screen_dist;

    public:
        camera() 
            : base(standard_base<N>()),
            screen_dims(from_value<N - 1>(DEFAULT_SCREEN_DIM)),
            screen_dist(DEFAULT_SCREEN_DIST) {}

        camera(const vector<N - 1>& screen_dims,
            const vec_type& screen_dist) 
            : base(standard_base<N>()),
            screen_dims(screen_dims),
            screen_dist(screen_dist) {}

    template <typename... Types,
        typename = std::enable_if_t<sizeof...(Types) == N && are_same<vector<N>, Types...>::value>>
        camera(const vector<N - 1>& screen_dims,
            const vec_type& screen_dist,
            Types... base_vecs)
            : screen_dims(screen_dims), 
            screen_dist(screen_dist) {
            base = {base_vecs...};
        }

        camera(const vector<N - 1>& screen_dims,
            const vec_type& screen_dist,
            vector<N>* arr)
            : base(arr),
            screen_dims(screen_dims),
            screen_dist(screen_dist) {}

        ray<N> get_ray(vec_type* screen_pos_arr) const {
            vector<N> screen_point;
            screen_point[0] = screen_dist;
            for (size_t i = 1; i < N; ++i)
                screen_point[i] = screen_pos_arr[i - 1] * screen_dims[i - 1];

            vector<N> scr_pt_cam_pov;
            for (size_t i = 0; i < N; ++i)
                scr_pt_cam_pov = scr_pt_cam_pov + (base[i] * screen_point[i]);

            return ray<N>(scr_pt_cam_pov, scr_pt_cam_pov - EPSILON);
        }

    template <typename... Types, 
        typename = std::enable_if_t<sizeof...(Types) == N - 1 && are_same<vec_type, Types...>::value>>
        ray<N> get_ray(const Types&... screen_pos) const {
            std::array<vec_type, N - 1> screen_pos_arr;
            screen_pos_arr = { static_cast<vec_type>(screen_pos)... };
            
            return get_ray(screen_pos_arr.data());
        }
    };
_RTC_END

#endif /* __CAMERA_H__ */
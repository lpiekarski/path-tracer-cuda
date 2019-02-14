#ifndef __TRACER_H__
#define __TRACER_H__

#include <memory>
#include <random>
#include <type_traits>
#ifdef RTC_TIMER
#ifdef RTC_USE_CUDA
#define init_sample()
#define add_sample()
#else
#include <chrono>
#include <iostream>

#define init_sample() \
    size_t samples = 0; \
    auto begin_clock = std::chrono::high_resolution_clock::now()

#define add_sample() \
    ++samples; \
    if ((samples & 8191) == 0) { \
        std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - begin_clock; \
        if (elapsed.count() > 1) { \
            std::cout << "samples per second: " << std::fixed << samples / elapsed.count() << std::endl; \
            begin_clock = std::chrono::high_resolution_clock::now(); \
            samples = 0; \
        } \
    }
#endif
#else
#define init_sample()
#define add_sample()
#endif 

#include "rtc.h"

#include "bitmap.h"
#include "camera.h"
#include "color.h"
#include "ray.h"
#include "tracable.h"

namespace rtc {

    constexpr size_t MAX_RAY_BOUNCES = 5;

    template <class TracableContainer,
        size_t _Dims,
        size_t BPP>
    class tracer {
    private:
        TracableContainer container;
        camera<_Dims> cam;

        _DEVHOST color<BPP> trace_ray_rec(const ray<_Dims>& r, size_t recursion_level) const {
            if (recursion_level >= MAX_RAY_BOUNCES)
                return color<BPP>();
            tracable<_Dims, BPP>* tr_obj = nullptr;
            vec_type dst;
            if (container.ray_intersection(r, &tr_obj, dst)) {
                vector<_Dims> intersection_point = r.get_point(dst);
                color<BPP> ambient = tr_obj->ambient_color(r, intersection_point);
                color<BPP> diffuse = tr_obj->diffuse_color(r, intersection_point);
                ray<_Dims> reflection_ray(tr_obj->get_reflection(r, intersection_point));
                color<BPP> reflection = trace_ray_rec(reflection_ray, recursion_level + 1);
                /*if (recursion_level >= 2) {
                    std::cout << recursion_level << std::endl;
                }*/
                return /*(reflection_ray.dir() + 1) / 2;*//*dot(r.dir(), vector<_Dims>() - reflection_ray.dir()) * */reflection * diffuse + ambient;
            }
            return color<BPP>();
        }

    public:
        using _Mytype = tracer<TracableContainer, _Dims, BPP>;

        _DEVHOST tracer() : container(), cam() {}

        _DEVHOST tracer(const tracer&) = delete;

        _DEVHOST tracer& operator=(const tracer&) = delete;

        _DEVHOST tracer(tracer&& other) : container(std::move(other.container)) {}

        _DEVHOST tracer& operator=(tracer&& other) {
            container = std::move(other.container);
            return *this;
        }

        _DEVHOST void add_tracable(tracable<_Dims, BPP>& t) {
            container.add(t);
        }

    template <typename... _Args,
        typename = enable_if_t<sizeof...(_Args) == _Dims - 1 && are_convertible<vec_type, _Args...>::value>>
        _DEVHOST color<BPP> trace_from_camera(const _Args&... screen_pos) const {
            return trace_ray(cam.get_ray(screen_pos...));
        }

        _DEVHOST color<BPP> trace_from_camera(vec_type* screen_pos_arr) const {
            return trace_ray(cam.get_ray(screen_pos_arr));
        }

        _DEVHOST color<BPP> trace_ray(const ray<_Dims>& r) const {
            return trace_ray_rec(r, 0);
        }
        
    template <size_t samples_per_pixel = 1,
        typename... _Args,
        typename = enable_if_t<_Dims >= 3 && sizeof...(_Args) == _Dims - 1 && are_convertible<size_t, _Args...>::value>>
        _HOST _vector<bitmap<BPP>> draw_bitmap(_Args... dims) {
            // init rng structures
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<vec_type> dist(0, 1);

            _array<size_t, _Dims - 1> dims_arr = { static_cast<size_t>(dims)... };
            size_t ret_len = dims_arr[0] * dims_arr[1];
            for (size_t i = 2; i < _Dims - 1; ++i)
                ret_len *= dims_arr[i];
            _vector<bitmap<BPP>> ret;
            init_sample();
            _array<vec_type, samples_per_pixel> random_vector;
            for (size_t smp = 0; smp < samples_per_pixel; ++smp)
                random_vector[smp] = dist(mt);

            // loop creating ret_len / dims_arr[0] / dims_arr[1] bitmaps
            for (size_t i = 0; i < ret_len / dims_arr[0] / dims_arr[1]; ++i) {
                bitmap<BPP> ret_bmp(dims_arr[0], dims_arr[1]);
                // rendering result bitmap pixel by pixel samples_per_pixel times
                for (size_t x = 0; x < dims_arr[0]; ++x) {
                    for (size_t y = 0; y < dims_arr[1]; ++y) {
                        color<BPP> sample_col;
                        for (size_t smp = 0; smp < samples_per_pixel; ++smp) {
                            _array<vec_type, _Dims - 1> screen_coord;
                            size_t c = i;
                            screen_coord[0] = (static_cast<vec_type>(x + random_vector[smp]) / dims_arr[0] - (vec_type)0.5) * (vec_type)2;
                            screen_coord[1] = (static_cast<vec_type>(y + random_vector[smp]) / dims_arr[1] - (vec_type)0.5) * (vec_type)2;
                            for (size_t j = 2; j < _Dims - 1; ++j) {
                                screen_coord[j] = (static_cast<vec_type>((c % dims_arr[j]) + random_vector[smp]) / dims_arr[j] - (vec_type)0.5) * (vec_type)2;
                                c /= dims_arr[j];
                            }
                            sample_col = sample_col + trace_from_camera(screen_coord.data()) / (vec_type)samples_per_pixel;
                            add_sample();
                        }
                        ret_bmp.set(x, y, sample_col);
                    }
                }
                ret.push_back(ret_bmp);
            }

            return ret;
        }

#ifdef RTC_USE_CUDA
    /*
    1 process for each bitmap
    1 thread for each pixel in bitmap

    */
    template <size_t samples_per_pixel = 1,
        typename... _Args,
        typename = enable_if_t<_Dims >= 3 && sizeof...(_Args) == _Dims - 1 && are_convertible<size_t, _Args...>::value>>
        _DEVICE _vector<bitmap<BPP>> device_draw_bitmap(_Args... dims) {
        //TODO
    }

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

#endif /* __TRACER_H__ */
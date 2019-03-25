#ifndef __DEFS_H__
#define __DEFS_H__

#ifdef __cplusplus
#define _RTC_BEGIN namespace rtc {
#define _RTC_END }
#else
#define _RTC_BEGIN
#define _RTC_END
#endif

#include "dh_array.h"
#include "dh_vector.h"

#ifdef RTC_USE_CUDA
#include <cuda_runtime.h>
#define _DEVICE __device__
#define _HOST __host__
#define _GLOBAL __global__
#define _DEVHOST _DEVICE _HOST
#else /* !RTC_USE_CUDA */
#define _DEVICE
#define _HOST
#define _DEVHOST
#define _GLOBAL
#endif

_RTC_BEGIN
template <typename _Ty>
    using _vector = dh_vector<_Ty>;

template <typename _Ty, size_t _Size>
    using _array = dh_array<_Ty, _Size>;
_RTC_END

#endif /* __DEFS_H__ */
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
#include <vector>
#include "dh_vector.h"

_RTC_BEGIN
#ifdef RTC_USE_CUDA
#include <cuda_runtime.h>
#define _DEVICE __device__
#define _HOST __host__
#define _DEVHOST _DEVICE _HOST
#else /* !RTC_USE_CUDA */
#define _DEVICE
#define _HOST
#define _DEVHOST
#endif

template <typename _Ty>
    using _vector = std::vector<_Ty>;//dh_vector<_Ty>;

template <typename _Ty, size_t _Size>
    using _array = dh_array<_Ty, _Size>;
_RTC_END

#endif /* __DEFS_H__ */
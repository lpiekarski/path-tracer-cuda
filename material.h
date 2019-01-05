#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <memory>

#include "rtc.h"
#include "bitmap.h"
//TODO: replace shared pointers with c style pointers
_RTC_BEGIN
    // CLASS TEMPLATE material
template <size_t BPP>
    class material {
    private:
        bitmap<BPP> *diffuse,
            *ambient,
            *reflection,
            *refraction, 
            *normal;
            //subsurface
    public:
        _DEVHOST material() 
            : diffuse(nullptr),
            ambient(nullptr),
            reflection(nullptr),
            refraction(nullptr),
            normal(nullptr) {}
    };
_RTC_END

#endif /* __MATERIAL_H__ */
#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <memory>

#include "rtc.h"
#include "bitmap.h"

_RTC_BEGIN
    // CLASS TEMPLATE material
template <size_t BPP>
    class material {
    private:
        std::shared_ptr<bitmap<BPP>> diffuse, ambient, reflection, refraction, normal;
    public:
    };
_RTC_END

#endif /* __MATERIAL_H__ */
#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include <array>
#include <memory>

#include "rtc.h"
#include "tracable.h"
#include "vector.h"

// triangle data:
// - shared 3 vertices
// - shared 3 normal vectors
// - shared material
// - diffuse map coords
// - normal map coords
// - reflection map coords
// - refraction map coords
// - ambient map coords
_RTC_BEGIN
template <size_t N,
    size_t BPP>
    class triangle : public tracable<N, BPP> {
    private:
        std::array<std::shared_ptr<vector<N>>, 3> vertices;
        color<BPP> ambient, diffuse;
        //TODO: array of normal vectors' shared pointers

        normal<N> get_normal() {
            normal<N> n = cross((*vertices[1]) - (*vertices[0]), (*vertices[2]) - (*vertices[0]));
            return n;
        }

    public:
        triangle() 
            : tracable<N, BPP>(),
            ambient(color<BPP>()),
            diffuse(white<BPP>()) {}

        triangle(std::shared_ptr<vector<N>> v1,
            std::shared_ptr<vector<N>> v2,
            std::shared_ptr<vector<N>> v3) 
            : tracable<N, BPP>(),
            ambient(color<BPP>()),
            diffuse(white<BPP>()) {
            vertices = { v1, v2, v3 };
        }

        triangle(const color<BPP>& ambient,
            const color<BPP>& diffuse)
            : tracable<N, BPP>(),
            ambient(ambient),
            diffuse(diffuse) {}

        triangle(std::shared_ptr<vector<N>> v1,
            std::shared_ptr<vector<N>> v2,
            std::shared_ptr<vector<N>> v3,
            const color<BPP>& ambient,
            const color<BPP>& diffuse) 
            : tracable<N, BPP>(),
            ambient(ambient),
            diffuse(diffuse) {
            vertices = { v1, v2, v3 };
        }

        triangle(const vector<N>& v1,
            const vector<N>& v2,
            const vector<N>& v3,
            const color<BPP>& ambient,
            const color<BPP>& diffuse) 
            : tracable<N, BPP>(),
            ambient(ambient),
            diffuse(diffuse) {
            vertices = { 
                std::make_shared<vector<N>>(v1),
                std::make_shared<vector<N>>(v2),
                std::make_shared<vector<N>>(v3) 
            };
        }

        color<BPP> ambient_color(const ray<N>& r,
            const vector<N>& intersection_point) {
            return ambient;
        }

        color<BPP> diffuse_color(const ray<N>& r,
            const vector<N>& intersection_point) {
            return diffuse;
        }

        ray<N> get_reflection(const ray<N>& r,
            const vector<N>& intersection_point) {
            normal<N> n(get_normal());
            if (dot(n, r.dir()) > 0)
                n = vector<N>() - n;
            ray<N> ret(reflect(r.dir(), n), intersection_point);
            return ray<N>(ret.dir(), ret.get_point(SURFACE_EPSILON));
        }

        bool ray_intersection(const ray<N>& r,
            std::conditional_t<N != 3, vec_type&, void*> intersection_dist) {
            throw std::exception("unimplemented function");
        }

        bool ray_intersection(const ray<N>& r,
            std::conditional_t<N == 3, vec_type&, void*> intersection_dist) {
            vec3 vertex0 = *vertices[0];
            vec3 vertex1 = *vertices[1];
            vec3 vertex2 = *vertices[2];
            vec3 edge1, edge2, h, s, q;
            vec_type a, f, u, v;
            edge1 = vertex1 - vertex0;
            edge2 = vertex2 - vertex0;
            h = cross(r.dir(), edge2);
            a = dot(edge1, h);
            if (a > -EPSILON && a < EPSILON)
                return false;    // This ray is parallel to this triangle.
            f = static_cast<vec_type>(1.0) / a;
            s = r.o() - vertex0;
            u = f * dot(s, h);
            if (u < 0.0 || u > 1.0)
                return false;
            q = cross(s, edge1);
            v = f * dot(r.dir(), q);
            if (v < 0.0 || u + v > 1.0)
                return false;
            // At this stage we can compute t to find out where the intersection point is on the line.
            vec_type t = f * dot(edge2, q);
            if (t > EPSILON) {
                intersection_dist = t;
                return true;
            }
            else // This means that there is a line intersection but not a ray intersection.
                return false;
        }
    };
_RTC_END

#endif /* __TRIANGLE_H__ */
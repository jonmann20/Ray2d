#ifndef VEC_H_
#define VEC_H_

#include <math.h>

/*
 * A 2D vector
 */
class Vec {
public:
	float x, y;

	Vec() {}

	Vec(float x, float y)
		: x(x), y(y)
	{}

	Vec operator+(const Vec& other) {
		return Vec(x + other.x, y + other.y);
	}

	Vec operator-(const Vec& other) {
		return Vec(x - other.x, y - other.y);
	}

	Vec operator/(const Vec& other) {
		return Vec(x / other.x, y / other.y);
	}

	// dot/inner product
	Vec operator*(const Vec& other) {
		return Vec(x * other.x, y * other.y);
	}

	float cross(const Vec& other) {
		return 0;		// z dimension always 0?
	}

	float length() {
		return sqrt(x*x + y*y);
	}
};

/*
* A 3D vector
*/
class Vec3 {
public:
	float x, y, z;

	Vec3() {}

	Vec3(float x, float y, float z)
		: x(x), y(y), z(z)
	{}
};

#endif // VEC_H
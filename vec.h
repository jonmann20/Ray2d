#ifndef VEC_H_
#define VEC_H_

#include <math.h>

/*
 * A 2D vector
 */
class Vec2 {
public:
	float x, y;

	Vec2() {}

	Vec2(float x, float y)
		: x(x), y(y)
	{}

	Vec2 operator+(const Vec2& other) {
		return Vec2(x + other.x, y + other.y);
	}

	Vec2 operator-(const Vec2& other) {
		return Vec2(x - other.x, y - other.y);
	}

	Vec2 operator/(const Vec2& other) {
		return Vec2(x / other.x, y / other.y);
	}

	// dot/inner product
	Vec2 operator*(const Vec2& other) {
		return Vec2(x * other.x, y * other.y);
	}

	float cross(const Vec2& other) {
		return x * other.y - y * other.x;
	}

	bool operator==(const Vec2& other) {
		return (abs(x - other.x) < 0.001) && (abs(y - other.y) < 0.001);
	}

	float length() {
		return sqrt(x*x + y*y);
	}
};

/*
* A 3D Vector
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
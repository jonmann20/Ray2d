#ifndef VEC_H_
#define VEC_H_

#include <math.h>

/*
 * A 2D vector
 */
class Vec2 {
public:
	float x, y;
	static const Vec2 ZERO;

	Vec2() {}
	Vec2(float x, float y);

	Vec2 operator+(const Vec2& other);
	Vec2 operator-(const Vec2& other);
	Vec2 operator/(const Vec2& other);
	Vec2 operator*(const Vec2& other);
		// dot/inner product

	float cross(const Vec2& other) const;

	bool operator==(const Vec2& other);
	bool operator!=(const Vec2& other);

	float length() const;
};


/*
* A 3D Vector
*/
class Vec3 {
public:
	float x, y, z;

	Vec3() {}
	Vec3(float x, float y, float z);

	void multiply(float s);
		// scalar multiplication

	void add(float s);
};

#endif // VEC_H
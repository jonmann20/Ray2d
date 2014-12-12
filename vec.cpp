#include "vec.h"

#include <math.h>

/*
* A 2D vector
*/

const Vec2 Vec2::ZERO = Vec2(0, 0);

Vec2::Vec2(float x, float y)
	: x(x), y(y)
{}

Vec2 Vec2::operator+(const Vec2& other) {
	return Vec2(x + other.x, y + other.y);
}

Vec2 Vec2::operator-(const Vec2& other) {
	return Vec2(x - other.x, y - other.y);
}

Vec2 Vec2::operator/(const Vec2& other) {
	return Vec2(x / other.x, y / other.y);
}

Vec2 Vec2::operator*(const Vec2& other) {
	return Vec2(x * other.x, y * other.y);
}

float Vec2::cross(const Vec2& other) const {
	return x * other.y - y * other.x;
}

bool Vec2::operator==(const Vec2& other) {
	return (abs(x - other.x) < 0.001) && (abs(y - other.y) < 0.001);
}

bool Vec2::operator!=(const Vec2& other) {
	return (abs(x - other.x) > 0.001) || (abs(y - other.y) > 0.001);
}

float Vec2::length() const {
	return sqrt(x*x + y*y);
}


/*
* A 3D Vector
*/

Vec3::Vec3(float x, float y, float z)
	: x(x), y(y), z(z)
{}

void Vec3::multiply(float s) {
	x *= s;
	y *= s;
	z *= s;
}

void Vec3::add(float s) {
	x += s;
	y += s;
	z += s;
}
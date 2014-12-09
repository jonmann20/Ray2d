#ifndef CIRCLE_H_
#define CIRCLE_H_

#include "vec.h"

class Circle {
public:
	Vec2 pos;
	float r;

	Circle() {}

	Circle(float x, float y, float radius)
		: pos(Vec2(x, y)), r(radius)
	{}
};

#endif // CIRCLE_H
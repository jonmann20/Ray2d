#ifndef CIRCLE_H_
#define CIRCLE_H_

#include "vec.h"

class Circle {
public:
	Vec pos;
	float r;

	Circle() {}

	Circle(float x, float y, float radius) {
		pos.x = x;
		pos.y = y;

		r = radius;
	}
};

#endif // CIRCLE_H
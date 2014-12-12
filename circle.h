#ifndef CIRCLE_H_
#define CIRCLE_H_

#include "vec.h"

class Circle {
public:
	Vec2 pos;
	float r;

	Circle(float x, float y, float radius);

	//void draw(Vec2 pos);
};

#endif // CIRCLE_H
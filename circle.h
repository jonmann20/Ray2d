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

	//void drawCircle(Vec2 pos) {
	//	glColor3f(1, 1, 0.5);
	//	glBegin(GL_POLYGON);
	//	double radius = 0.08;
	//	for(double i = 0; i < 2 * PI; i += PI / 24) { //<-- Change this Value
	//		glVertex2f(pos.x + (cos(i) * radius), pos.y + (sin(i) * 1.6 * radius));
	//	}
	//	glEnd();
	//}

};

#endif // CIRCLE_H
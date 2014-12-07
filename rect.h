#ifndef RECT_H_
#define RECT_H_

#include "vec.h"

class Rect {
public:
	Vec pos, size;
	Vec3 color;

	Rect() {}

	Rect(float x, float y, float w, float h, Vec3 color) {
		pos.x = x;
		pos.y = y;
		
		size.x = w;
		size.y = h;

		this->color = color;
	}
};

#endif // RECT_H
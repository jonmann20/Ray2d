#ifndef RECT_H_
#define RECT_H_

#include "vec.h"

class Rect {
public:
	Vec pos, size;

	Rect() {}

	Rect(float x, float y, float w, float h) {
		pos.x = x;
		pos.y = y;
		
		size.x = w;
		size.y = h;
	}
};

#endif // RECT_H
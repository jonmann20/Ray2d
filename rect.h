#ifndef RECT_H_
#define RECT_H_

#include "vec.h"

class Rect {
public:
	Vec2 pos, size;
	Vec3 INIT_COLOR;
	Vec3 color;

	Rect(float x, float y, float w, float h);

	void draw() const;
};

#endif // RECT_H
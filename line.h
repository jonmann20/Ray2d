#ifndef LINE_H_
#define LINE_H_

#include "vec.h"

class Line {
public:
	Vec2 start, end;

	Line() {}

	Line(float x1, float y1, float x2, float y2)
		: start(Vec2(x1, y1)), end(Vec2(x2, y2))
	{}
};

#endif // LINE_H
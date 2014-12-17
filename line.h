#ifndef LINE_H_
#define LINE_H_

#include "utils.h"
#include "vec.h"

class Line {
public:
	Vec2 start, end;
	DirType type;

	Line(float x1, float y1, float x2, float y2, DirType type = DirType::NONE);

	Vec2 midPt();
	float length();
		// Distance Formula
};

#endif // LINE_H
#ifndef LINE_H_
#define LINE_H_

#include "utils.h"
#include "vec.h"

class Line {
public:
	Vec2 start, end;
	DirType type;

	Line(float x1, float y1, float x2, float y2, DirType type = DirType::NONE);

	void draw() const;
		// EFFECTS: draws a line from start to end on the screen

	Vec2 midPt();
	float length();
		// Distance Formula
};

#endif // LINE_H
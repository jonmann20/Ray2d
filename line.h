#ifndef LINE_H_
#define LINE_H_

#include "vec.h"
#include "references.cu"

class Line {
public:
	Vec2 start, end;

	Line() {}

	Line(float x1, float y1, float x2, float y2)
		: start(Vec2(x1, y1)), end(Vec2(x2, y2))
	{}

	void draw() {
		glColor3f(0.8, 0, 0);
		glBegin(GL_LINES);
			glVertex2f(start.x, start.y);
			glVertex2f(end.x, end.y);
		glEnd();
	}
};

#endif // LINE_H
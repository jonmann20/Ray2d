#include "line.h"

#include "utils.h"
#include "vec.h"
#include "color.h"

#include <GL/glut.h>

Line::Line(float x1, float y1, float x2, float y2, DirType type /*= DirType::NONE*/)
	: start(Vec2(x1, y1)), end(Vec2(x2, y2)), type(type) 
{}

void Line::draw() const {
	glColor3f(Color::RED.x, Color::RED.y, Color::RED.z);
	glBegin(GL_LINES);
		glVertex2f(start.x, start.y);
		glVertex2f(end.x, end.y);
	glEnd();
}

Vec2 Line::midPt() {
	return Vec2((start.x + end.x) / 2, (start.y + end.y) / 2);
}

float Line::length() {
	return sqrt(pow(end.x - start.x, 2) + pow(end.y - start.y, 2));
}

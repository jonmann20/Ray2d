#include "rect.h"

#include "vec.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

Rect::Rect(float x, float y, float w, float h)
	: pos(Vec2(x, y)), size(Vec2(w, h))
{}

void Rect::draw() const {
	glColor3f(color.x, color.y, color.z);
	glBegin(GL_POLYGON);
	glVertex2f(pos.x, pos.y);
	glVertex2f(pos.x + size.x, pos.y);
	glVertex2f(pos.x + size.x, pos.y - size.y);
	glVertex2f(pos.x, pos.y - size.y);
	glEnd();
}
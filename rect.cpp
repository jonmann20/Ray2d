#include "rect.h"

#include "vec.h"

Rect::Rect(float x, float y, float w, float h)
	: pos(Vec2(x, y)), size(Vec2(w, h))
{}
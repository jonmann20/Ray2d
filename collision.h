#ifndef COLLISION_H_
#define COLLISION_H_

#include "vec.h"
#include "rect.h"
#include "line.h"

struct CollisionResponse {
	Vec2 intersectionPt;
	bool wasCollision;
};

CollisionResponse testLineLine(const Line& a, const Line& b);
	// http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/1968345#1968345

CollisionResponse testRectRect(Rect a, Rect b);

#endif // COLLISION_H
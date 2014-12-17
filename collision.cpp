#include "collision.h"

#include "vec.h"
#include "rect.h"
#include "line.h"

CollisionResponse testLineLine(const Line& a, const Line& b) {
	CollisionResponse cr;

	float p0_x = a.start.x;
	float p0_y = a.start.y;
	float p1_x = a.end.x;
	float p1_y = a.end.y;
	float p2_x = b.start.x;
	float p2_y = b.start.y;
	float p3_x = b.end.x;
	float p3_y = b.end.y;

	float s1_x, s1_y, s2_x, s2_y;
	s1_x = p1_x - p0_x;     s1_y = p1_y - p0_y;
	s2_x = p3_x - p2_x;     s2_y = p3_y - p2_y;

	float s, t;
	s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y);
	t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y);

	if(s >= 0 && s <= 1 && t >= 0 && t <= 1) {
		cr.intersectionPt.x = p0_x + (t * s1_x);
		cr.intersectionPt.y = p0_y + (t * s1_y);

		cr.wasCollision = true;
		return cr;
	}

	cr.wasCollision = false;
	return cr;
}

CollisionResponse testRectRect(Rect a, Rect b) {
	CollisionResponse cr;

	if(a.pos.x < (b.pos.x + b.size.x) && (a.pos.x + a.size.x) > b.pos.x &&
	   a.pos.y > (b.pos.y - b.size.y) && (a.pos.y - a.size.y) < b.pos.y
	) {
		// TODO: intersetionPt

		cr.wasCollision = true;
		return cr;
	}

	cr.wasCollision = false;
	return cr;
}
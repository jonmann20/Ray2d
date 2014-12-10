#ifndef COLLISION_H_
#define COLLISION_H_

//#include <iostream>

#include "vec.h"
#include "circle.h"
#include "rect.h"
#include "line.h"

//using namespace std;

struct CollisionResponse {
	Vec2 intersectionPt;
	bool wasCollision;
};

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
		if(cr.intersectionPt.x != NULL)
			cr.intersectionPt.x = p0_x + (t * s1_x);
		if(cr.intersectionPt.y != NULL)
			cr.intersectionPt.y = p0_y + (t * s1_y);

		cr.wasCollision = true;
		return cr;
	}

	cr.wasCollision = false;
	return cr;
}

CollisionResponse testRectRect(Rect a, Rect b) {
	CollisionResponse r;
	//r.overlapV = Vec2(0, 0);


	if(a.pos.x < (b.pos.x + b.size.x) && (a.pos.x + a.size.x) > b.pos.x &&
	   a.pos.y < (b.pos.y + b.size.y) && (a.pos.y + a.size.y) > b.pos.y
	) {
		//cout << "hit" << endl;
		//r.overlapV = Vec2(1, 1);
	}

	return r;
}

#endif // COLLISION_H



//bool testLineLine(const Line& a, const Line& b) {
//	Vec2 p = a.start;
//	Vec2 p2 = a.end;
//	Vec2 q = b.start;
//	Vec2 q2 = b.end;
//
//	Vec2 r = p2 - p;
//	Vec2 s = q2 - q;
//
//	float uNumerator = (q - p).cross(r);
//	float denominator = r.cross(s);
//
//	if(uNumerator == 0 && denominator == 0) {
//		// They are colinear
//
//		// Do they touch? (Are any of the points equal?)
//		if((p == q) || (p == q2) || (p2 == q) || (p2 == q2)) {
//			return true;
//		}
//		// Do they overlap? (Are all the point differences in either direction the same sign)
//		// Using != as exclusive or
//		return ((q.x - p.x < 0) != (q.x - p2.x < 0) != (q2.x - p.x < 0) != (q2.x - p2.x < 0)) ||
//			((q.y - p.y < 0) != (q.y - p2.y < 0) != (q2.y - p.y < 0) != (q2.y - p2.y < 0));
//	}
//
//	if(denominator == 0) {
//		// lines are paralell
//		return false;
//	}
//
//	float u = uNumerator / denominator;
//	float t = (q - p).cross(s) / denominator;
//
//	return (t >= 0) && (t <= 1) && (u >= 0) && (u <= 1);
//}
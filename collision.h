#ifndef COLLISION_H_
#define COLLISION_H_

//#include <iostream>

#include "vec.h"
#include "circle.h"
#include "rect.h"
#include "line.h"

//using namespace std;

struct CollisionResponse {
	Vec2 overlapN, overlapV;
};

bool testLineLine(const Line& a, const Line& b) {
	Vec2 p = a.start;
	Vec2 p2 = a.end;
	Vec2 q = b.start;
	Vec2 q2 = b.end;

	Vec2 r = p2 - p;
	Vec2 s = q2 - q;

	float uNumerator = (q - p).cross(r);
	float denominator = r.cross(s);

	if(uNumerator == 0 && denominator == 0) {
		// They are colinear

		// Do they touch? (Are any of the points equal?)
		if((p == q) || (p == q2) || (p2 == q) || (p2 == q2)) {
			return true;
		}
		// Do they overlap? (Are all the point differences in either direction the same sign)
		// Using != as exclusive or
		return ((q.x - p.x < 0) != (q.x - p2.x < 0) != (q2.x - p.x < 0) != (q2.x - p2.x < 0)) ||
			((q.y - p.y < 0) != (q.y - p2.y < 0) != (q2.y - p.y < 0) != (q2.y - p2.y < 0));
	}

	if(denominator == 0) {
		// lines are paralell
		return false;
	}

	float u = uNumerator / denominator;
	float t = (q - p).cross(s) / denominator;

	return (t >= 0) && (t <= 1) && (u >= 0) && (u <= 1);

}

CollisionResponse testRectRect(Rect a, Rect b) {
	CollisionResponse r;
	r.overlapV = Vec2(0, 0);


	if(a.pos.x < (b.pos.x + b.size.x) && (a.pos.x + a.size.x) > b.pos.x &&
	   a.pos.y < (b.pos.y + b.size.y) && (a.pos.y + a.size.y) > b.pos.y
	) {
		//cout << "hit" << endl;
		r.overlapV = Vec2(1, 1);
	}

	return r;
}

#endif // COLLISION_H
#ifndef LIGHT_H_
#define LIGHT_H_

#include "vec.h"
#include "line.h"

#include <omp.h>

#include <vector>
using namespace std;

enum LightType { FLOURESCENT, INCANDESCENT };

class Light {
private:
	const float INTENSITY;
	vector<Line> raySegments;
	vector<Line> rays;
	omp_lock_t raySegmentsLock;

public:
	Vec2 pos;
	Vec3 color;
	LightType type;
	bool raysVisible;
	
	Light(float x, float y, LightType type, Vec3 color, bool raysVisible=false);

	void updatePos();

	void checkRays();
	void checkRay(Line ray);
	void reflectRay(const Line& raySegment);
		// EFFECTS: calls checkRay recursively
		//			reflected ray's length is equalt to the raySegment, not the original ray

	void draw() const;
};

#endif // LIGHT_H
#ifndef LIGHT_H_
#define LIGHT_H_

#include "vec.h"
#include "line.h"

#include <vector>
using namespace std;

enum LightType { FLOURESCENT, INCANDESCENT };

class Light {
private:
	const float INTENSITY;
	vector<Line> raySegments;
	vector<Line> rays;

public:
	Vec2 pos;
	Vec3 color;
	LightType type;
	bool raysVisible;
	
	Light(float x, float y, LightType type, Vec3 color, bool raysVisible=false);

	void checkRays();
	void checkRay(Line ray);
	void draw();
};

#endif // LIGHT_H
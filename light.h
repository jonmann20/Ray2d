#ifndef LIGHT_H_
#define LIGHT_H_

#include "vec.h"

enum LightType { FLOURESCENT, INCANDESCENT };

class Light {
public:
	Vec2 pos;
	Vec3 color;
	LightType type;
	bool raysVisible;

	Light() {}

	Light(float x, float y, LightType type, Vec3 color, bool raysVisible = false) : type(type), color(color), raysVisible(raysVisible) {
		pos.x = x;
		pos.y = y;
	}
};

#endif // LIGHT_H
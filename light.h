#ifndef LIGHT_H_
#define LIGHT_H_

#include "vec.h"
#include "line.h"

enum LightType { FLOURESCENT, INCANDESCENT };

class Light {
public:
	Vec2 pos;
	Vec3 color;
	LightType type;
	bool raysVisible;
	vector<Line> rays;

	Light() {}

	Light(float x, float y, LightType type, Vec3 color, bool raysVisible = false)
		: pos(Vec2(x, y)), type(type), color(color), raysVisible(raysVisible)
	{
		Line line = Line(0, 0.85, 0, -1);
		rays.push_back(line);
	}

	void draw() {
		float offsetY = 0.08;

		glColor3f(color.x, color.y, color.z);
		glBegin(GL_POLYGON);
			glVertex2f(pos.x, pos.y);
			glVertex2f(pos.x + 0.08, pos.y + offsetY);
			glVertex2f(pos.x + 0.24, pos.y + offsetY);
			glVertex2f(pos.x + 0.32, pos.y);
		glEnd();

		if(raysVisible) {
			drawRays();
		}

		//glColor3f(0, 0, 0);
		//drawText(light.pos + Vec2(0.115, 0.025), "Spot Light");
	}

	void drawRays() {
		//if(!raysVisible) return;

		for(auto ray : rays) {
			ray.draw();
		}
	}
};

#endif // LIGHT_H
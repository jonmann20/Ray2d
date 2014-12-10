#ifndef LIGHT_H_
#define LIGHT_H_

#include "globals.h"
#include "line.h"
#include "player.h"
#include "collision.h"

enum LightType { FLOURESCENT, INCANDESCENT };

class Light {
public:
	Vec2 pos;
	Vec3 color;
	LightType type;
	bool raysVisible;
	vector<Line> rays;
	vector<Line> raySegments;

	Light() {}

	Light(float x, float y, LightType type, Vec3 color, bool raysVisible = false)
		: pos(Vec2(x, y)), type(type), color(color), raysVisible(raysVisible)
	{
		Line line = Line(0.1, 0.85, -0.3, -0.7);
		rays.push_back(line);
	}

	void checkRays() {
		// reset player color
		for(auto& i : player.body) {
			i.color = i.INIT_COLOR;
		}

		// reset raySegments
		raySegments.clear();

		for(const auto& ray : rays) {
			checkRay(ray);
		}
	}

	void checkRay(Line ray) {
		Line raySegment = ray;

		for(auto& chunk : player.body) {
			float x = player.pos.x + chunk.pos.x;
			float y = player.pos.y + chunk.pos.y;
			float x2 = x + chunk.size.x;
			float y2 = y + chunk.size.y;

			Line chunkLines[4] = {
				Line(x, y, x2, y),
				Line(x2, y, x2, y2),
				Line(x2, y2, x, y2),
				Line(x, y2, x, y)
			};

			CollisionResponse cr;
			for(int i = 0; i < 4; ++i) {
				CollisionResponse cr = testLineLine(ray, chunkLines[i]);

				if(cr.wasCollision) {
					chunk.color = Vec3(1, 1, 1);

					raySegment.end = cr.intersectionPt;

					// spawn ray reflection
					Line newRay = Line(
						cr.intersectionPt.x,
						cr.intersectionPt.y,
						(ray.end.x - ray.start.x) * 2,
						ray.start.y - ray.end.y
					);
					//checkRay(newRay);
					
					raySegments.push_back(raySegment);
					return;
				}
			}
		}

		raySegments.push_back(raySegment);
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
			for(const auto& i : raySegments) {
				i.draw();
			}
		}
	}
};

#endif // LIGHT_H
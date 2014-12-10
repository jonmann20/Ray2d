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

	Light() {}

	Light(float x, float y, LightType type, Vec3 color, bool raysVisible = false)
		: pos(Vec2(x, y)), type(type), color(color), raysVisible(raysVisible)
	{
		Line line = Line(0.1, 0.85, -0.3, -0.7);
		rays.push_back(line);
	}

	void checkRays() {
		vector<Line> newRays;
		
		// reset player color
		for(auto& i : player.body) {
			i.color = i.INIT_COLOR;
		}

		for(const auto& ray : rays) {
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

				for(int i=0; i < 4; ++i) {
					if(testLineLine(ray, chunkLines[i])) {
						// TODO: only push once?
						// spawn new ray
						newRays.push_back(Line(
							chunkLines[i].midPt().x,
							chunkLines[i].midPt().y,
							ray.end.x - ray.start.x,
							ray.start.y - ray.end.y
						));

						chunk.color = Vec3(1, 1, 1);
						break;
					}
				}
			}
		}

		for(const auto& i : newRays) {
			rays.push_back(i);
		}
		newRays.clear();
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
	}

	void drawRays() {
		for(const auto& ray : rays) {
			ray.draw();
		}
	}
};

#endif // LIGHT_H
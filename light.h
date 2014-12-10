#ifndef LIGHT_H_
#define LIGHT_H_

#include "globals.h"
#include "line.h"
#include "player.h"
#include "collision.h"
#include "chunk.h"

enum LightType { FLOURESCENT, INCANDESCENT };

int stackCount = 0;

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
		//printV(stackCount);

		Line raySegment = ray;

		for(auto& chunk : player.body) {
			for(auto& line : chunk.lines) {
				// convert to absolute world position
				float x = player.pos.x + line.start.x;
				float y = player.pos.y + line.start.y;
				float x2 = player.pos.x + line.end.x;
				float y2 = player.pos.y + line.end.y;
				

				CollisionResponse cr = testLineLine(ray, Line(x, y, x2, y2, line.type));

				if(cr.wasCollision && ray.start != cr.intersectionPt) {
					chunk.color = Vec3(1, 1, 1);

					raySegment.end = cr.intersectionPt;

					Vec2 endPt;
					if(line.type == LineType::TOP) {
						endPt = Vec2((ray.end.x - ray.start.x) * 2, ray.start.y - ray.end.y);
					}
					else if(line.type == LineType::RIGHT) {
						endPt = Vec2(ray.start.x, -ray.start.y);
					}
					else if(line.type == LineType::BOT) {
						endPt = Vec2(-1, 1);
					}
					else {					// left
						endPt = Vec2(1, 1);
					}

					// spawn ray reflection
					Line newRay = Line(
						cr.intersectionPt.x,
						cr.intersectionPt.y,
						endPt.x,
						endPt.y
					);

					//++stackCount;

					raySegments.push_back(raySegment);
					return checkRay(newRay);
				}
			}



			/*float x = player.pos.x + chunk.rect.pos.x;
			float y = player.pos.y + chunk.rect.pos.y;
			float x2 = x + chunk.size.x;
			float y2 = y - chunk.size.y;

			Line chunkLines[4] = {
				Line(x, y, x2, y),
				Line(x2, y, x2, y2),
				Line(x2, y2, x, y2),
				Line(x, y2, x, y)
			};*/

			//CollisionResponse cr;
			//for(int i = 0; i < 4; ++i) {
			//	CollisionResponse cr = testLineLine(ray, chunkLines[i]);

			//	if(cr.wasCollision && ray.start != cr.intersectionPt) {
			//		chunk.color = Vec3(1, 1, 1);

			//		raySegment.end = cr.intersectionPt;

			//		Vec2 endPt;
			//		if(i == 0) {			// top
			//			endPt = Vec2((ray.end.x - ray.start.x) * 2, ray.start.y - ray.end.y);
			//		}
			//		else if(i == 1) {		// right
			//			endPt = Vec2(ray.start.x, -ray.start.y);
			//		}
			//		else if(i == 2) {		// bottom
			//			endPt = Vec2(-1, 1);
			//		}
			//		else {					// left
			//			endPt = Vec2(1, 1);
			//		}

			//		// spawn ray reflection
			//		Line newRay = Line(
			//			cr.intersectionPt.x,
			//			cr.intersectionPt.y,
			//			endPt.x,
			//			endPt.y
			//		);
			//		
			//		++stackCount;

			//		raySegments.push_back(raySegment);
			//		return checkRay(newRay);
			//	}
			//}
		}

		//--stackCount;
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
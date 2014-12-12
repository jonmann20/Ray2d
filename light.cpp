#include "light.h"

#include "game.h"
#include "vec.h"
#include "line.h"
#include "utils.h"
#include "player.h"
#include "collision.h"
#include "chunk.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
using namespace std;


Light::Light(float x, float y, LightType type, Vec3 color, bool raysVisible /*=false*/)
	: INTENSITY(0.1), pos(Vec2(x, y)), type(type), color(color), raysVisible(raysVisible)
{
	for(int i=0; i < 50; ++i) {
		rays.push_back(Line(0.1, 0.85, -0.35 + i*0.005, -0.7, DirType::NONE));
	}
}

void Light::checkRays() {
	// reset player color
	for(auto& chunk : player.body) {
		chunk.color = chunk.INIT_COLOR;
	}

	// reset raySegments
	raySegments.clear();

	for(const auto& ray : rays) {
		checkRay(ray);
	}
}

void Light::checkRay(Line ray) {
	Line raySegment = ray;
	int chunkNum = 0;

	//// window edges
	//Line window[4] ={
	//	Line(0, 0, game.FULLW, 0),			// top
	//	Line(game.FULLW, 0, game.FULLW, game.FULLH),	// right
	//	Line(game.FULLW, game.FULLH, 0, game.FULLH),	// bot
	//	Line(0, game.FULLH, 0, 0)			// left
	//};

	//for(auto edge : window) {
	//	CollisionResponse cr = testLineLine(ray, edge);
	//	if(cr.wasCollision && ray.start != cr.intersectionPt) {
	//		raySegment.end = cr.intersectionPt;
	//		raySegments.push_back(raySegment);

	//		return checkRay(Line(cr.intersectionPt.x, cr.intersectionPt.y, 0, 0));	// TODO: calc endPt
	//	}
	//}

	// player
	for(auto& chunk : player.body) {
		for(auto& line : chunk.lines) {
			// convert to absolute world position
			float x = player.pos.x + line.start.x;
			float y = player.pos.y + line.start.y;
			float x2 = player.pos.x + line.end.x;
			float y2 = player.pos.y + line.end.y;
			
			CollisionResponse cr = testLineLine(ray, Line(x, y, x2, y2));

			if(cr.wasCollision && ray.start != cr.intersectionPt) {
				player.updateChunkColors(chunkNum, INTENSITY);
				
				raySegment.end = cr.intersectionPt;
				raySegments.push_back(raySegment);

				// TODO: use trig with angles
				// calculate newRay's endPt
				Vec2 endPt;
				if(line.type == DirType::TOP) {
					endPt = Vec2((ray.end.x - ray.start.x) * 2, ray.start.y - ray.end.y);
				}
				else if(line.type == DirType::RIGHT) {
					endPt = Vec2(ray.start.x, -ray.start.y);
				}
				else if(line.type == DirType::BOT) {
					endPt = Vec2(-1, 1);
				}
				else {					// left
					endPt = Vec2(1, 1);
				}

				// recurse on reflected ray
				return checkRay(Line(cr.intersectionPt.x, cr.intersectionPt.y, endPt.x, endPt.y));
			}
		}

		++chunkNum;
	}

	raySegments.push_back(raySegment);
}

void Light::draw() const {
	// light
	float offsetY = 0.08;

	glColor3f(color.x, color.y, color.z);
	glBegin(GL_POLYGON);
		glVertex2f(pos.x, pos.y);
		glVertex2f(pos.x + 0.08, pos.y + offsetY);
		glVertex2f(pos.x + 0.24, pos.y + offsetY);
		glVertex2f(pos.x + 0.32, pos.y);
	glEnd();

	// rays
	if(raysVisible) {
		for(const auto& i : raySegments) {
			i.draw();
		}
	}
}

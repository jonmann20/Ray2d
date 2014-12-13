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
{}

void Light::checkRays() {
	// reset player color
	for(auto& chunk : player.body) {
		chunk.color = chunk.INIT_COLOR;
	}

	// reset raySegments
	raySegments.clear();

	// Shoot out rays
	const int NUM_RAYS = 46;
	const float OFFSETX = 0.04;
	const float WIDTH = 0.32;
	const float DTX = (WIDTH - OFFSETX*2) / NUM_RAYS;
	const float DTX2 = (WIDTH + OFFSETX*4) / NUM_RAYS;

	for(int i=0; i < NUM_RAYS; ++i) {
		checkRay(Line(pos.x + OFFSETX + i*DTX, pos.y, pos.x - OFFSETX*2 + i*DTX2, -0.8, DirType::NONE));
	}
}

void Light::reflectRay(const Line& raySegment) {
	float dx = raySegment.start.x - raySegment.end.x;
	float dy = raySegment.start.y - raySegment.end.y;
	float theta = atan2f(dy, dx) * 180/PI;	// counter clockwise
	float x, y;
	
	if(theta == 0 || theta == 90 || theta == 180 || theta == 270 || theta == 360) {
		return;
	}

	switch(raySegment.type) {
		case DirType::TOP:
			x = raySegment.end.x - dx;
			y = raySegment.start.y;
			break;
		case DirType::RIGHT:
			//if(theta < 90) {
				y = raySegment.end.y - dy;
			//}
			//else {
				//y = raySegment.end.y + dy;
			//}

			x = raySegment.start.x;
			break;
		case DirType::BOT:
			// TODO:
			break;
		case DirType::LEFT:
			y = raySegment.end.y - dy;
			x = raySegment.start.x;
			break;
	}

	//pr3nt(theta, x, y);

	return checkRay(Line(raySegment.end.x, raySegment.end.y, x, y));
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
	for(const auto& chunk : player.body) {
		for(const auto& line : chunk.lines) {
			// convert to absolute world position
			float x = player.pos.x + line.start.x;
			float y = player.pos.y + line.start.y;
			float x2 = player.pos.x + line.end.x;
			float y2 = player.pos.y + line.end.y;
			
			CollisionResponse cr = testLineLine(ray, Line(x, y, x2, y2));

			if(cr.wasCollision && ray.start != cr.intersectionPt) {
				player.updateChunkColors(chunkNum, INTENSITY);

				raySegment.end = cr.intersectionPt;
				raySegment.type = line.type;
				raySegments.push_back(raySegment);

				return reflectRay(raySegment);
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

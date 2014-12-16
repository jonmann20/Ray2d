#include "light.h"

#include "game.h"
#include "vec.h"
#include "line.h"
#include "utils.h"
#include "player.h"
#include "collision.h"
#include "chunk.h"
#include "color.h"

#include <omp.h>

#include <GL/glut.h>

#include <iostream>
using namespace std;

Light::Light(float x, float y, LightType type, Vec3 color, bool raysVisible /*=false*/)
	: INTENSITY(0.1), pos(Vec2(x, y)), type(type), /*color(color),*/ raysVisible(raysVisible)
{
	this->color = Vec3(1, 1, 1);
	printxyz(color);
}

void Light::checkRays() {
	
	// reset player color
	//profiler.start();
	for(auto& chunk : player.body) {
		chunk.color = chunk.INIT_COLOR;
	}
	//profiler.end("reset player color");

	// reset raySegments
	//profiler.start();
	raySegments.clear();
	//profiler.end("reset raySegments");

	// Shoot out rays
	//profiler.start();
	const int SPREAD_FACTOR = 16;
	const int NUM_RAYS = 23;//46;
	const float OFFSETX = 0.04;
	const float WIDTH = 0.32;
	const float DTX = (WIDTH - OFFSETX*2) / NUM_RAYS;
	const float DTX2 = (WIDTH + OFFSETX*(SPREAD_FACTOR*2)) / NUM_RAYS;

	for(int i=0; i < NUM_RAYS; ++i) {
		Line ray = Line(pos.x + OFFSETX + i*DTX, pos.y, pos.x - OFFSETX*SPREAD_FACTOR + i*DTX2, -0.8);
		checkRay(ray);
	}

	//checkRay(Line(pos.x, pos.y, pos.x + OFFSETX*5, -0.8));
	//profiler.end("ray colliison");
	//cin.get();
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
			x = raySegment.start.x;
			y = raySegment.end.y - dy;
			break;
		case DirType::BOT:
			// TODO:
			break;
		case DirType::LEFT:
			y = raySegment.end.y - dy;
			x = raySegment.start.x;
			break;
	}

	return checkRay(Line(raySegment.end.x, raySegment.end.y, x, y));
}

void Light::checkRay(Line ray) {
	Line raySegment = ray;

	for(int j=0; j < player.body.size(); ++j) {
		for(int k=0; k < player.body[j].lines.size(); ++k) {
			Line line = player.body[j].lines[k];

			// convert to absolute world position
			const float x = player.pos.x + line.start.x;
			const float y = player.pos.y + line.start.y;
			const float x2 = player.pos.x + line.end.x;
			const float y2 = player.pos.y + line.end.y;

			CollisionResponse cr = testLineLine(ray, Line(x, y, x2, y2));

			if(cr.wasCollision && ray.start != cr.intersectionPt) {
				//profiler.start();
				player.updateChunkColors(j, INTENSITY);
				//profiler.endAvg("omp updateChunks");

				raySegment.end = cr.intersectionPt;
				raySegment.type = line.type;

				raySegments.push_back(raySegment);
				
				return reflectRay(raySegment);
			}
		}
	}

	raySegments.push_back(raySegment);
}

void Light::draw() const {
	// light
	const float offsetY = 0.08;
	
	printxyz(color);

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
#include "light.h"

#include "vec.h"
#include "line.h"
#include "utils.h"
#include "player.h"
#include "collision.h"
#include "chunk.h"
#include "profiler.h"
#include "input.h"

#include <omp.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
using namespace std;

Light::Light(float x, float y, LightType type, Vec3 color, bool raysVisible /*=false*/)
	: INTENSITY(0.1), pos(Vec2(x, y)), type(type), color(color), raysVisible(raysVisible)
{
	omp_init_lock(&raySegmentsLock);
}

void Light::updatePos() {
	const float DT = 0.05;

	if(keysDown['i']) {
		pos.y += DT;
	}

	if(keysDown['j']) {
		pos.x -= DT;
	}

	if(keysDown['k']) {
		pos.y -= DT;
	}

	if(keysDown['l']) {
		pos.x += DT;
	}
}

void Light::checkRays() {
	// reset player color
#pragma omp parallel for
	for(int i=0; i < player.body.size(); ++i) {
		player.body[i].color = player.body[i].INIT_COLOR;
	}

	// reset raySegments
	raySegments.clear();

	// Shoot out rays
	//profiler.start();
	const int SPREAD_FACTOR = 16;
	const int NUM_RAYS = 10;//64;
	const float OFFSETX = 0.04;
	const float WIDTH = 0.32;
	const float DTX = (WIDTH - OFFSETX*2) / NUM_RAYS;
	const float DTX2 = (WIDTH + OFFSETX*(SPREAD_FACTOR*2)) / NUM_RAYS;

	//#pragma omp parallel for
	for(int i=0; i < NUM_RAYS; ++i) {
		Line ray = Line(pos.x + OFFSETX + i*DTX, pos.y, pos.x - OFFSETX*SPREAD_FACTOR + i*DTX2, pos.y - 1.5);
		checkRay(ray);
	}
	//profiler.end("ray colliison");
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
			x = raySegment.end.x - dx;
			y = raySegment.start.y;
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

	// window edges
	Line window[4] = {
		Line(-1, 1, 1, 1, DirType::BOT),	// top
		Line(1, 1, 1, -1, DirType::LEFT),	// right
		Line(1, -1, -1, -1, DirType::TOP),	// bot
		Line(-1, -1, -1, 1, DirType::RIGHT)	// left
	};

	for(const auto& edge : window) {
		CollisionResponse cr = testLineLine(ray, edge);
		if(cr.wasCollision && ray.start != cr.intersectionPt) {
			raySegment.end = cr.intersectionPt;
			raySegment.type = edge.type;
			
			omp_set_lock(&raySegmentsLock);
			raySegments.push_back(raySegment);
			omp_unset_lock(&raySegmentsLock);

			return reflectRay(raySegment);
		}
	}

	for(int j=0; j < player.body.size(); ++j) {
		for(int k=0; k < player.body[j].lines.size(); ++k) {
			Line line = player.body[j].lines[k];

			// convert to absolute world position
			const float x = player.pos.x + line.start.x;
			const float y = player.pos.y + line.start.y;
			const float x2 = player.pos.x + line.end.x;
			const float y2 = player.pos.y + line.end.y;

			// fix line order (e.g. ray from below hitting bottom of top line before checking bottom line)
			if(((ray.start.y >= line.start.y) && (ray.start.y >= line.end.y) && (line.type == DirType::BOT))   ||	// top line
			   ((ray.start.y <= line.start.y) && (ray.start.y <= line.end.y) && (line.type == DirType::TOP))   ||	// bot line
			   ((ray.start.x <= line.start.x) && (ray.start.x <= line.end.x) && (line.type == DirType::RIGHT)) ||	// left line
			   ((ray.start.x >= line.start.x) && (ray.start.x >= line.end.x) && (line.type == DirType::LEFT))		// right line
			) {
				// do nothing
			}
			else {
				CollisionResponse cr = testLineLine(ray, Line(x, y, x2, y2));

				if(cr.wasCollision && ray.start != cr.intersectionPt) {
					//profiler.start();
					player.updateChunkColors(j, INTENSITY);
					//profiler.end("updateChunks");

					raySegment.end = cr.intersectionPt;
					raySegment.type = line.type;

					omp_set_lock(&raySegmentsLock);
					raySegments.push_back(raySegment);
					omp_unset_lock(&raySegmentsLock);

					return reflectRay(raySegment);
				}
			}
		}
	}

	omp_set_lock(&raySegmentsLock);
	raySegments.push_back(raySegment);
	omp_unset_lock(&raySegmentsLock);
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

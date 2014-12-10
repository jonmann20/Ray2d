#ifndef PLAYER_H_
#define PLAYER_H_

#include "references.cu"
#include <vector>
#include "rect.h"
#include "chunk.h"
#include "input.h"

/*
 * The user controllable character of the game.
 */
class Player {
public:
	Vec2 pos;
	vector<Chunk> body;		// relative to pos
	
	Player() {}

	Player(float x, float y, float w, float h) {
		pos = Vec2(x, y);
		body.push_back(Chunk(0, 0, w / 2, h / 2, Vec3(1, 0, 0), TOP_LEFT));				// red
		body.push_back(Chunk(w / 2, 0, w / 2, h / 2, Vec3(0, 1, 0), TOP_RIGHT));		// green
		body.push_back(Chunk(w / 2, -h / 2, w / 2, h / 2, Vec3(0, 0, 1), BOT_RIGHT));	// blue
		body.push_back(Chunk(0, -h / 2, w / 2, h / 2, Vec3(1, 1, 0.5), BOT_LEFT));		// yellow
	}

	// Draws a rectangle in chunks
	void draw() {
		for(auto part : body) {
			float x = pos.x + part.rect.pos.x;
			float y = pos.y + part.rect.pos.y;

			glColor3f(part.color.x, part.color.y, part.color.z);

			glBegin(GL_POLYGON);
			glVertex2f(x, y);
			glVertex2f(x + part.rect.size.x, y);
			glVertex2f(x + part.rect.size.x, y - part.rect.size.y);
			glVertex2f(x, y - part.rect.size.y);
			glEnd();
		}
	}

	void updatePos() {
		const float dt = 0.004;

		if(keysDown['w']) {
			pos.y += dt;
		}
		if(keysDown['a']) {
			pos.x -= dt;
		}
		if(keysDown['s']) {
			pos.y -= dt;
		}
		if(keysDown['d']) {
			pos.x += dt;
		}
		//if(input.keysDown[32]) {			// spacebar
		//	debugRays = !debugRays;
		//}
		if(keysDown[27]) {			// escape
			exit(0);
		}

		glutPostRedisplay();
	}
} player;

#endif // PLAYER_H
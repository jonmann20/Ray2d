#ifndef PLAYER_H_
#define PLAYER_H_

#include <vector>
#include "rect.h"
#include "references.cu"
#include "input.h"

/*
 * The user controllable character of the game.
 */
class Player {
public:
	Vec2 pos;
	vector<Rect> body;		// relative to pos
	
	Player() {}

	Player(float x, float y, float w, float h) {
		pos = Vec2(x, y);
		body.push_back(Rect(0, 0, w/2, h/2, Vec3(1, 0, 0)));		// top left; red
		body.push_back(Rect(w/2, 0, w/2, h/2, Vec3(0, 1, 0)));		// top right; green
		body.push_back(Rect(w/2, -h/2, w/2, h/2, Vec3(0, 0, 1)));		// bot right; blue
		body.push_back(Rect(0, -h/2, w/2, h/2, Vec3(1, 1, 0.5)));		// bot left; yellow
	}

	// Draws a rectangle in chunks
	void draw() {
		for(auto part : body) {
			float x = pos.x + part.pos.x;
			float y = pos.y + part.pos.y;

			glColor3f(part.color.x, part.color.y, part.color.z);

			glBegin(GL_POLYGON);
			glVertex2f(x, y);
			glVertex2f(x + part.size.x, y);
			glVertex2f(x + part.size.x, y + part.size.y);
			glVertex2f(x, y + part.size.y);
			glEnd();
		}


		//glColor3f(0, 0, 0);
		//drawText(pos + Vec2(-0.025, -0.01), "Player");
	}

	void updatePos() {
		const float dt = 0.04;

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
};

#endif // PLAYER_H
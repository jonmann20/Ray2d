#ifndef PLAYER_H_
#define PLAYER_H_

#include <vector>
#include "rect.h"

/*
 * The main character of the game.
 */
class Player {

private:
	Vec pos;

public:
	vector<Rect> body;		// relative to pos
	
	Player() {}

	Player(float x, float y, float w, float h) {
		pos = Vec(x, y);
		body.push_back(Rect(0, 0, w/2, h/2, Vec3(1, 0, 0)));		// top left; red
		body.push_back(Rect(w/2, 0, w/2, h/2, Vec3(0, 1, 0)));		// top right; green
		body.push_back(Rect(w/2, -h/2, w/2, h/2, Vec3(0, 0, 1)));		// bot right; blue
		body.push_back(Rect(0, -h/2, w/2, h/2, Vec3(1, 1, 0.5)));		// bot left; yellow
	}

	Vec& getPos() {
		return pos;
	}
};

#endif // PLAYER_H
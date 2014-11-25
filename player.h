#ifndef PLAYER_H_
#define PLAYER_H_

#include "circle.h"

/*
 * The main character of the game.
 */
class Player {
public:
	Circle body;

	Player() {}

	Player(float x, float y, float r) {
		body.pos.x = x;
		body.pos.y = y;
		body.r = r;
	}

	Vec& getPos() {
		return body.pos;
	}
};

#endif // PLAYER_H
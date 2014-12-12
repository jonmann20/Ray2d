#ifndef GAME_H_
#define GAME_H_

class Vec2;
class Light;

#include <vector>
using namespace std;

/*
 * Holds game engine state and utility classes.
 */
class Game {
private:
	// fps
	int frameCount;
	float fps;
	int currentTime, previousTime;

public:
	const int FULLW, FULLH;
	vector<Light> lights;

	// debugging
	const float DEBUG_INFOX;

	Game();

	void drawLights();
	void checkRayCollision();

	void calculateFPS();
	void drawFPS();
	void drawText(Vec2 pos, char* format, ...);
};

extern Game game;

#endif // GAME_H
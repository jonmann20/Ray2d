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

	// debugging
	const float DEBUG_INFOX;

public:
	const int FULLW, FULLH;
	vector<Light> lights;

	Game();

	void drawLights() const;
	void checkRayCollision();

	void calculateFPS();
	void drawFPS() const;
	void drawText(Vec2 pos, char* format, ...) const;
};

extern Game game;

#endif // GAME_H
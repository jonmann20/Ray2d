#ifndef GLOBALS_H_
#define GLOBALS_H_

#include "vec.h"
#include "player.h"

// player
Player player;

// fps
int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;

// debug
bool debugRays = true;
const float DEBUG_INFOX = -0.98;


#endif // GLOBALS_H
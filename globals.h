#ifndef GLOBALS_H_
#define GLOBALS_H_

#include "light.h"

#include <vector>

using namespace std;

// Game Objects
vector<Light> lights;

// Game Engine components


// fps
int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;

// debug
bool debugRays = true;
const float DEBUG_INFOX = -0.98;


#endif // GLOBALS_H
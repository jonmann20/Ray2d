#include "input.h"
//#include "utils.h"

bool keysDown[256];

void keydown(unsigned char key, int x, int y) {
	//printn(key);
	keysDown[key] = true;
}

void keyup(unsigned char key, int x, int y) {
	keysDown[key] = false;
}
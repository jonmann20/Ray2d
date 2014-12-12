#include "game.h"

//#include "references.cu"
//#include "utils.h"
#include "vec.h"
#include "light.h"

#include <vector>
using namespace std;

Game game;

Game::Game() :
	FULLW(720), FULLH(720),
	frameCount(0), fps(0), currentTime(0), previousTime(0),
	debugRays(true), DEBUG_INFOX(-0.98)//,
	//lights({Light(12)})
	//fontStyle(GLUT_BITMAP_HELVETICA_12),
	//lights({Light(0, 0.85, LightType::FLOURESCENT, Vec3(1, 0.95686, 0.89804), true)})		// http://planetpixelemporium.com/tutorialpages/light.html
{}
/*
void Game::calculateFPS() {
	++frameCount;

	// Get the number of milliseconds since glutInit called 
	// (or first call to glutGet(GLUT ELAPSED TIME)).
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	// Calculate time passed
	int timeInterval = currentTime - previousTime;

	if(timeInterval > 1000) {
		// calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);

		// Set time
		previousTime = currentTime;

		// Reset frame count
		frameCount = 0;
	}
}

void Game::drawFPS() {
	//  Load the identity matrix so that FPS string being drawn won't get animates
	glLoadIdentity();

	glColor3f(0.6, 0.6, 0);
	drawText(Vec2(DEBUG_INFOX, 0.92), "FPS: %4.2f", fps);
}

void Game::drawText(Vec2 pos, char* format, ...) {
	// Initialize a variable argument list
	va_list args;
	va_start(args, format);

	// Return the number of characters in the string referenced the list of arguments.
	// _vscprintf doesn't count terminating '\0' (that's why +1)
	int len = _vscprintf(format, args) + 1;

	// Allocate memory for a string of the specified size
	char* text = (char*)malloc(len * sizeof(char));

	// Write formatted output using a pointer to the list of arguments
	vsprintf_s(text, len, format, args);

	// End using variable argument list 
	va_end(args);

	// Specify the raster position for pixel operations
	glRasterPos2f(pos.x, pos.y);

	for(int i=0; text[i] != '\0'; ++i) {
		//glutBitmapCharacter(fontStyle, text[i]);
	}

	free(text);
}*/
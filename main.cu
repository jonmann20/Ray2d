// OpenGL, OpenGl Utitliy Toolkit (GLUT)
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// FlappyRay helpers
#include "utils.h"			// includes: iostream, namespace std
#include "globals.h"		// includes: light
#include "rect.h"			// includes: vec2
#include "circle.h"			// includes: vec2
#include "line.h"			// includes: vec2
#include "input.h"
#include "collision.h"
#include "player.h"

// OpenGL globals
GLvoid* font_style = GLUT_BITMAP_HELVETICA_12;

#pragma region CUDA
__global__
void square(int* a) {
	*a = (*a) * (*a);
}


void testCuda() {
	int a = 8;
	int* da;	// device copy
	int size = sizeof(int);

	// Init da on GPU
	cudaMalloc((void**)&da, size);
	cudaMemcpy(da, &a, size, cudaMemcpyHostToDevice);

	square << <1, 1 >> >(da);

	// Grab answer from GPU
	cudaMemcpy(&a, da, size, cudaMemcpyDeviceToHost);
	cudaFree(da);

	cout << "a: " << a << endl;
}
#pragma endregion CUDA

#pragma region Update
void checkRayCollision() {
	for(auto& light : lights) {
		light.checkRays();
	}
}

void update() {
	player.updatePos();
	checkRayCollision();

	glutPostRedisplay();
}

void calculateFPS() {
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
#pragma endregion Update

#pragma region Render
void drawText(Vec2 pos, char* format, ...) {
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
		glutBitmapCharacter(font_style, text[i]);
	}

	free(text);
}

void drawFPS() {
	//  Load the identity matrix so that FPS string being drawn won't get animates
	glLoadIdentity();

	glColor3f(0.6, 0.6, 0);
	drawText(Vec2(DEBUG_INFOX, 0.92), "FPS: %4.2f", fps);
}

void drawLights() {
	for(auto light : lights) {
		light.draw();
	}
}

void render() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Level
	drawLights();

	// Player
	player.draw();

	// Debug
	calculateFPS();
	drawFPS();

	//if(debugRays) {
	//	drawRays();
	//}
	//else {
	//	glColor3f(0.6, 0.6, 0);
	//	drawText(Vec2(DEBUG_INFOX, 0.87), "DebugRays Off");
	//}

	glutSwapBuffers();
}
#pragma endregion Render


int main(int argc, char* argv[]) {
	//----- Game Setup
	player = Player(-0.05, 0, 0.25, 0.25);

	Vec3 warmFlourescent = Vec3(1, 0.95686, 0.89804);		// http://planetpixelemporium.com/tutorialpages/light.html
	lights.push_back(Light(0, 0.85, LightType::FLOURESCENT, warmFlourescent, true));


	//----- OpenGL setup
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(720, 720);		// 1280 x 720
	glutInitWindowPosition(800, 160);
	glutCreateWindow("FlappyRay Engine Demo");

	//glutGameModeString("1280x720:16@60");		// 16 bits per pixel
	//glutEnterGameMode();

	glutDisplayFunc(render);
	glutIdleFunc(update);
	//glutTimerFunc(32, update, -1);

	glutIgnoreKeyRepeat(1);
	glutKeyboardFunc(keydown);
	glutKeyboardUpFunc(keyup);
	//glutSpecialFunc(keyboard);

	glutMainLoop();


	return EXIT_SUCCESS;
}
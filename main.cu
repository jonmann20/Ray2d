// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// FlappyRay helpers
#include "utils.h"			// includes: iostream, namespace std
#include "globals.h"		// includes: vec, player
#include "rect.h"			// includes: vec
#include "circle.h"			// includes: vec

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
void keyboard(unsigned char key, int x, int y) {
	printV(key);

	switch(key) {
		case 'w':
			player.getPos().y += 0.1;
			break;
		case 'a':
			player.getPos().x -= 0.1;
			break;
		case 's':
			player.getPos().y -= 0.1;
			break;
		case 'd':
			player.getPos().x += 0.1;
			break;
		case 32:			// spacebar
			debugRays = !debugRays;
			break;
		case 27:			// escape
			exit(0);
			break;
	}
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


void update() {
	calculateFPS();
}

#pragma endregion Update

#pragma region Render
void drawText(Vec pos, char* format, ...) {
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
	drawText(Vec(DEBUG_INFOX, 0.92), "FPS: %4.2f", fps);
}

void drawRect(Vec pos) {
	glBegin(GL_POLYGON);
		glVertex2f(pos.x, pos.y);
		glVertex2f(pos.x + 0.5, pos.y);
		glVertex2f(pos.x + 0.5, pos.y + 0.5);
		glVertex2f(pos.x, pos.y + 0.5);
	glEnd();
}

void drawCircle(Vec pos) {
	glColor3f(1, 1, 0.5);
	glBegin(GL_POLYGON);
		double radius = 0.08;
		for(double i = 0; i < 2 * PI; i += PI / 24) { //<-- Change this Value
			glVertex2f(pos.x + (cos(i) * radius), pos.y + (sin(i) * 1.6 * radius));
		}
	glEnd();

	glColor3f(0, 0, 0);
	drawText(pos + Vec(-0.025, -0.01), "Player");
}

void drawLight(Vec pos) {
	float offsetY = 0.08;

	glColor3f(1, 0.95686, 0.89804);		// warm flourescent (http://planetpixelemporium.com/tutorialpages/light.html)
	glBegin(GL_POLYGON);
		glVertex2f(pos.x, pos.y);
		glVertex2f(pos.x + 0.08, pos.y + offsetY);
		glVertex2f(pos.x + 0.24, pos.y + offsetY);
		glVertex2f(pos.x + 0.32, pos.y);
	glEnd();

	glColor3f(0, 0, 0);
	drawText(pos + Vec(0.115, 0.025), "Spot Light");
}

void drawRays() {
	glColor3f(0.6, 0.6, 0);
	drawText(Vec(DEBUG_INFOX, 0.87), "DebugRays On");

	Vec pos(0, 0.85);
	Vec size(0.32, 0.08);

	glColor3f(0.8, 0, 0);
	glBegin(GL_LINES);
		glVertex2f(pos.x + size.x/2, pos.y);
		glVertex2f(pos.x + size.x/2, pos.y - (2 - (1-pos.y)));
	glEnd();
}

void render() {
	glClear(GL_COLOR_BUFFER_BIT);

	// Level
	drawLight(Vec(0, 0.85));

	// Player
	drawCircle(player.getPos());

	// Debug
	drawFPS();

	if(debugRays) {
		drawRays();
	}
	else {
		glColor3f(0.6, 0.6, 0);
		drawText(Vec(DEBUG_INFOX, 0.87), "DebugRays Off");
	}

	glFlush();
}
#pragma endregion Render

//void gameLoop() {
//	update();
//	render();
//}

int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(1280, 720);
	glutInitWindowPosition(320, 180);
	glutCreateWindow("FlappyRay Engine Demo");
	glutDisplayFunc(render);
	glutIdleFunc(update);

	glutKeyboardFunc(keyboard);
	//glutSpecialFunc(keyboard);

	glutMainLoop();

	return EXIT_SUCCESS;
}
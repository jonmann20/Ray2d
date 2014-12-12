// OpenGL, OpenGl Utitliy Toolkit (GLUT)
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// FlappyRay helpers
#include "game.h"
#include "light.h"
//#include "utils.h"
//#include "rect.h"
//#include "circle.h"
//#include "line.h"
//#include "input.h"
//#include "collision.h"
//#include "player.h"


#include <iostream>
using namespace std;

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
	/*for(Light light : game.lights) {
		light.checkRays();
	}*/
}

void update() {
	//player.updatePos();
	checkRayCollision();

	glutPostRedisplay();
}


#pragma endregion Update

#pragma region Render


void drawLights() {
	/*for(auto light : game.lights) {
		light.draw();
	}*/

	glColor3f(0.8, 0, 0);
	glBegin(GL_LINES);
	glVertex2f(0, 0);
	glVertex2f(-1, 1);
	glEnd();
}

void render() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Level
	drawLights();

	// Player
	//player.draw();

	// Debug
	//game.calculateFPS();
	//game.drawFPS();

	//if(game.debugRays) {
	//	drawRays();
	//}
	//else {
	//	glColor3f(0.6, 0.6, 0);
	//	drawText(Vec2(game.DEBUG_INFOX, 0.87), "DebugRays Off");
	//}

	glutSwapBuffers();
}
#pragma endregion Render


int main(int argc, char* argv[]) {
	//----- Game Setup
	//player = Player(0.1, 0, 0.25, 0.25);

	//----- OpenGL setup
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(game.FULLW, game.FULLH);
	glutInitWindowPosition(850, 160);
	glutCreateWindow("FlappyRay Engine Demo");

	glutDisplayFunc(render);
	glutIdleFunc(update);
	//glutTimerFunc(32, update, -1);

	glutIgnoreKeyRepeat(1);
	//glutKeyboardFunc(keydown);
	//glutKeyboardUpFunc(keyup);
	

	glutMainLoop();


	return EXIT_SUCCESS;
}
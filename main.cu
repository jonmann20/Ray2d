// OpenGL, OpenGl Utitliy Toolkit (GLUT)
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// FlappyRay helpers
#include "game.h"
#include "input.h"
#include "player.h"

//#include <iostream>
//using namespace std;

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

	//cout << "a: " << a << endl;
}
#pragma endregion CUDA

void update() {
	player.updatePos();
	game.checkRayCollision();

	glutPostRedisplay();
}

void render() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Level
	game.drawLights();

	// Player
	player.draw();

	// FPS
	game.calculateFPS();
	game.drawFPS();

	glutSwapBuffers();
}


int main(int argc, char* argv[]) {
	// OpenGL setup
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(game.FULLW, game.FULLH);
	glutInitWindowPosition(550, 160);
	glutCreateWindow("FlappyRay Engine Demo");

	glutDisplayFunc(render);
	glutIdleFunc(update);
	//glutTimerFunc(32, update, -1);

	glutIgnoreKeyRepeat(1);
	glutKeyboardFunc(keydown);
	glutKeyboardUpFunc(keyup);

	glutMainLoop();

	return EXIT_SUCCESS;
}
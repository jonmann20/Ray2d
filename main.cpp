#include <GL/glut.h>

#include "game.h"
#include "input.h"
#include "player.h"

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

	// // FPS
	// game.calculateFPS();
	// game.drawFPS();

	glutSwapBuffers();
}


int main(int argc, char* argv[]) {
	// OpenGL setup
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(game.FULLW, game.FULLH);
	glutInitWindowPosition(700, 40);
	glutCreateWindow("FlappyRay Engine Demo");

	glutDisplayFunc(render);
	glutIdleFunc(update);

	glutIgnoreKeyRepeat(1);
	glutKeyboardFunc(keydown);
	glutKeyboardUpFunc(keyup);

	glutMainLoop();

	return EXIT_SUCCESS;
}
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
	//printV(key);

	const float dt = 0.08;

	switch(key) {
		case 'w':
			player.pos.y += dt;
			break;
		case 'a':
			player.pos.x -= dt;
			break;
		case 's':
			player.pos.y -= dt;
			break;
		case 'd':
			player.pos.x += dt;
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


void checkRayCollision() {
	for(const auto& light : lights) {
		
		for(auto& chunk : player.body) {
		//auto chunk = player.body.back();
		
			float x = player.pos.x + chunk.pos.x;
			//float y = player.pos.y + chunk.pos.y;
		
			//cout << x << " <= " << light.pos.x << " && " << (x + chunk.size.x) << " >= " << light.pos.x << endl;

			if((x <= light.pos.x) &&
			   ((x + chunk.size.x) >= light.pos.x)
			){
				//cout << "new color" << endl;
				chunk.color = Vec3(1, 1, 1);
			}
			else {
				//cout << "init color" << endl;
				chunk.color = chunk.INIT_COLOR;
			}
		}
	}
}

void update() {
	checkRayCollision();

	glutPostRedisplay();
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

// Draws a rectangle in chunks
void drawRect(Vec2 pos, vector<Rect> body) {
	for(auto part : body) {
		float x = pos.x + part.pos.x;
		float y = pos.y + part.pos.y;

		glColor3f(part.color.x, part.color.y, part.color.z);

		glBegin(GL_POLYGON);
			glVertex2f(x, y);
			glVertex2f(x + part.size.x, y);
			glVertex2f(x + part.size.x, y + part.size.y);
			glVertex2f(x, y + part.size.y);
		glEnd();
	}


	//glColor3f(0, 0, 0);
	//drawText(pos + Vec2(-0.025, -0.01), "Player");
}

void drawCircle(Vec2 pos) {
	glColor3f(1, 1, 0.5);
	glBegin(GL_POLYGON);
		double radius = 0.08;
		for(double i = 0; i < 2 * PI; i += PI / 24) { //<-- Change this Value
			glVertex2f(pos.x + (cos(i) * radius), pos.y + (sin(i) * 1.6 * radius));
		}
	glEnd();
}

void drawRays(Light light) {
	//glColor3f(0.6, 0.6, 0);
	//drawText(Vec2(DEBUG_INFOX, 0.87), "DebugRays On");

	glColor3f(0.8, 0, 0);
	glBegin(GL_LINES);
	glVertex2f(light.pos.x, light.pos.y);
	glVertex2f(light.pos.x, light.pos.y - (2 - (1 - light.pos.y)));
	glEnd();
}

void drawLight(Light light) {
	float offsetY = 0.08;

	glColor3f(light.color.x, light.color.y, light.color.z);
	glBegin(GL_POLYGON);
		glVertex2f(light.pos.x, light.pos.y);
		glVertex2f(light.pos.x + 0.08, light.pos.y + offsetY);
		glVertex2f(light.pos.x + 0.24, light.pos.y + offsetY);
		glVertex2f(light.pos.x + 0.32, light.pos.y);
	glEnd();

	if(light.raysVisible) {
		drawRays(light);
	}

	//glColor3f(0, 0, 0);
	//drawText(light.pos + Vec2(0.115, 0.025), "Spot Light");
}

void drawLights() {
	for(auto light : lights) {
		drawLight(light);
	}
}

void render() {
	glClear(GL_COLOR_BUFFER_BIT);

	// Level
	drawLights();

	// Player
	drawRect(player.pos, player.body);

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

	glFlush();
}
#pragma endregion Render


int main(int argc, char* argv[]) {
	player = Player(0, 0, 0.2, 0.2);

	Vec3 warmFlourescent = Vec3(1, 0.95686, 0.89804);		// http://planetpixelemporium.com/tutorialpages/light.html
	lights.push_back(Light(0, 0.85, LightType::FLOURESCENT, warmFlourescent, true));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(720, 720);		// 1280 x 720
	glutInitWindowPosition(320, 180);
	glutCreateWindow("FlappyRay Engine Demo");
	

	//glutGameModeString("1280x720:16@60");		// 16 bits per pixel
	//glutEnterGameMode();


	glutDisplayFunc(render);
	glutIdleFunc(update);

	//glutTimerFunc(32, update, -1);

	glutKeyboardFunc(keyboard);
	//glutSpecialFunc(keyboard);

	glutMainLoop();

	return EXIT_SUCCESS;
}
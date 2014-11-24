// System 
#include <iostream>
#include <windows.h>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;

#define printV(v) cout << #v << ": " << v << endl

class Vec {
public:
	float x, y;

	Vec() {
	
	}
	
	Vec(float xx, float yy) {
		x = xx;
		y = yy;
	}
};

Vec box1pos;

int frameCount = 0;
float fps = 0;
int currentTime = 0, previousTime = 0;

GLvoid *font_style = GLUT_BITMAP_HELVETICA_18;//GLUT_BITMAP_TIMES_ROMAN_24;

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
//void input(unsigned char key, int xmouse, int ymouse) {
//	switch(key) {
//		case 'w':
//			box1pos.y += 0.1;
//			break;
//		case 'a':
//			break;
//		case 's':
//			box1pos.y -= 0.1;
//			break;
//		case 'd':
//			break;
//	}
//
//	printV(box1pos.x);
//	printV(box1pos.y);
//}

void keyboard(int key, int x, int y) {
	printV(key);
	switch(key) {
		case 100:		// left arrow
			box1pos.x -= 0.1;
			break;
		case 101:		// up arrow
			box1pos.y += 0.1;
			break;
		case 102:		// right arrow
			box1pos.x += 0.1;
			break;
		case 103:		// down arrow
			box1pos.y -= 0.1;
			break;
	}
	glutPostRedisplay();
}

void calculateFPS() {
	//  Increase frame count
	frameCount++;

	//  Get the number of milliseconds since glutInit called 
	//  (or first call to glutGet(GLUT ELAPSED TIME)).
	currentTime = glutGet(GLUT_ELAPSED_TIME);

	//  Calculate time passed
	int timeInterval = currentTime - previousTime;

	if(timeInterval > 1000) {
		//  calculate the number of frames per second
		fps = frameCount / (timeInterval / 1000.0f);

		//  Set time
		previousTime = currentTime;

		//  Reset frame count
		frameCount = 0;
	}
}


void update() {
	calculateFPS();
}

#pragma endregion Update

#pragma region Render
void printw(float x, float y, float z, char* format, ...) {
	va_list args;	//  Variable argument list
	int len;		//	String length
	int i;			//  Iterator
	char * text;	//	Text

	//  Initialize a variable argument list
	va_start(args, format);

	//  Return the number of characters in the string referenced the list of arguments.
	//  _vscprintf doesn't count terminating '\0' (that's why +1)
	len = _vscprintf(format, args) + 1;

	//  Allocate memory for a string of the specified size
	text = (char *)malloc(len * sizeof(char));

	//  Write formatted output using a pointer to the list of arguments
	vsprintf_s(text, len, format, args);

	//  End using variable argument list 
	va_end(args);

	//  Specify the raster position for pixel operations.
	glRasterPos3f(x, y, z);

	//  Draw the characters one by one
	for(i = 0; text[i] != '\0'; i++)
		glutBitmapCharacter(font_style, text[i]);

	//  Free the allocated memory for the string
	free(text);
}

void drawFPS() {
	//  Load the identity matrix so that FPS string being drawn
	//  won't get animates
	glLoadIdentity();

	//  Print the FPS to the window
	printw(0.77, 0.9, 0, "FPS: %4.2f", fps);
}

void drawRect(Vec pos) {
	glBegin(GL_POLYGON);
	glVertex3f(pos.x, pos.y, 0.0);
	glVertex3f(pos.x + 0.5, pos.y, 0.0);
	glVertex3f(pos.x + 0.5, pos.y + 0.5, 0.0);
	glVertex3f(pos.x, pos.y + 0.5, 0.0);
	glEnd();
}

void render() {
	glClear(GL_COLOR_BUFFER_BIT);

	drawFPS();
	drawRect(box1pos);

	glFlush();
}
#pragma endregion Render

void gameLoop() {
	update();
	render();
}

int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(1280, 720);
	glutInitWindowPosition(320, 180);
	glutCreateWindow("FlappyRay Engine Demo");
	glutDisplayFunc(render);
	glutIdleFunc(update);

	//glutKeyboardFunc(input);
	glutSpecialFunc(keyboard);

	glutMainLoop();

	cin.get();
	return EXIT_SUCCESS;
}

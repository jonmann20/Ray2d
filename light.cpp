#include "light.h"

#include "game.h"
#include "vec.h"
//#include "line.h"
//#include "player.h"
//#include "collision.h"
//#include "chunk.h"

#include <iostream>
using namespace std;


Light::Light(float x, float y, LightType type, Vec3 color, bool raysVisible)	// raysVisible = false
	//: pos(Vec2(x, y)), type(type), color(color), raysVisible(raysVisible)
{


	cout << "In Light(int f)" << endl;
	//cout << "f: " << f << endl;
	cout << "FULLW: " << game.FULLW << endl;
}
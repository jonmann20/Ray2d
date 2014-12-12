#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
using namespace std;

#define PI 3.1415926535897932384626433832795
#define printV(v) cout << #v << ": " << v << endl
#define printV2(a, b) cout << #a << ", " << #b << ": " << a << ", " << b << endl

enum DirType { TOP, RIGHT, BOT, LEFT, NONE };
const char* const DirTypes[] = {"TOP", "RIGHT", "BOT", "LEFT", "NONE"};


//Vec3(1, 0, 0)		// red
//Vec3(0, 1, 0)		// green
//Vec3(0, 0, 1)		// blue
//Vec3(1, 1, 0.5)	// yellow

#endif // UTILS_H
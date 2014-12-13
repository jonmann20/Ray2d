#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
using namespace std;

#define PI 3.1415926535897932384626433832795
#define pr1nt(v)		cout << #v << ": " << v << endl
#define pr2nt(a, b)		cout << '(' << #a << ", " << #b << "): (" << a << ", " << b << ")\n"
#define pr3nt(a, b, c)	cout << '(' << #a << ", " << #b << ", " <<  #c << "): (" << a << ", " << b << ", " << c << ")\n"
//#define printN(...) cout << ... << endl;

enum DirType { TOP, RIGHT, BOT, LEFT, NONE };
const char* const DirTypes[] = {"TOP", "RIGHT", "BOT", "LEFT", "NONE"};


//Vec3(1, 0, 0)		// red
//Vec3(0, 1, 0)		// green
//Vec3(0, 0, 1)		// blue
//Vec3(1, 1, 0.5)	// yellow

#endif // UTILS_H
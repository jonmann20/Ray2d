#ifndef UTILS_H_
#define UTILS_H_

#include <stdlib.h>
#include <stdio.h>

#include <iostream>
using namespace std;

#define PI 3.1415926535897932384626433832795

enum DirType { TOP, RIGHT, BOT, LEFT, NONE };
const char* const DirTypes[] = {"TOP", "RIGHT", "BOT", "LEFT", "NONE"};

//Vec3(1, 0, 0)		// red
//Vec3(0, 1, 0)		// green
//Vec3(0, 0, 1)		// blue
//Vec3(1, 1, 0.5)	// yellow

//----- Pretty print objects with special members
#define printxy(obj) cout << #obj << ": (" << obj.x << ", " << obj.y << ")\n"
#define printxyz(obj) cout << #obj << ": (" << obj.x << ", " << obj.y << ", " << obj.z << ")\n"

//----- Pretty print n generic variables
// Macro overloading
#define EXPAND(X) X
#define PP_NARG(...) EXPAND( PP_NARG_(__VA_ARGS__, PP_RSEQ_N()) )
#define PP_NARG_(...) EXPAND( PP_ARG_N(__VA_ARGS__) )
#define PP_ARG_N(_1, _2, _3, _4, N, ...) N
#define PP_RSEQ_N() 4, 3, 2, 1, 0

// printn
#define printn_(N) printn_##N
#define printn_EVAL(N) printn_(N)
#define printn(...) EXPAND( printn_EVAL(EXPAND( PP_NARG(__VA_ARGS__) ))(__VA_ARGS__) )

#define printn_1(a)				cout << #a << ": " << a << endl
#define printn_2(a, b)			cout << '(' << #a << ", " << #b << "): (" << a << ", " << b << ")\n"
#define printn_3(a, b, c)		cout << '(' << #a << ", " << #b << ", " <<  #c << "): (" << a << ", " << b << ", " << c << ")\n"
#define printn_4(a, b, c, d)	cout << '(' << #a << ", " << #b << ", " <<  #c <<  << #d << "): (" << a << ", " << b << ", " << c << ", " << d << ")\n"

#endif // UTILS_H
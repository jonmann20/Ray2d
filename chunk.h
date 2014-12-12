#ifndef CHUNK_H_
#define CHUNK_H_

#include "vec.h"
#include "line.h"
#include "rect.h"

#include <vector>
using namespace std;

enum ChunkType {TOP_LEFT, TOP_MID, TOP_RIGHT, MID_LEFT, MID, MID_RIGHT, BOT_LEFT, BOT_MID, BOT_RIGHT};
const char* const ChunkTypes[] = {"TOP_LEFT", "TOP_MID", "TOP_RIGHT", "MID_LEFT", "MID", "MID_RIGHT", "BOT_LEFT", "BOT_MID", "BOT_RIGHT"};

class Chunk {
public:
	Rect rect;
	Vec3 INIT_COLOR, color;
	vector<Line> lines;
	ChunkType type;

	Chunk(float x, float y, float w, float h, Vec3 color, ChunkType type);
};

#endif // RECT_H
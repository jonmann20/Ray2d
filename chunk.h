#ifndef CHUNK_H_
#define CHUNK_H_

#include "vec.h"
#include "rect.h"

enum ChunkType {TOP_LEFT, TOP_MID, TOP_RIGHT, MID_LEFT, MID, MID_RIGHT, BOT_LEFT, BOT_MID, BOT_RIGHT};
char* ChunkTypes[] = {"TOP_LEFT", "TOP_MID", "TOP_RIGHT", "MID_LEFT", "MID", "MID_RIGHT", "BOT_LEFT", "BOT_MID", "BOT_RIGHT"};

class Chunk {
public:
	Rect rect;
	Vec3 INIT_COLOR, color;
	vector<Line> lines;
	ChunkType type;

	Chunk() {}

	Chunk(float x, float y, float w, float h, Vec3 color, ChunkType type)
		: rect(Rect(x, y, w, h)), INIT_COLOR(color), color(color), type(type)
	{
		// All lines (for debugging)
		/*lines.push_back(Line(x, y, x+w, y, DirType::TOP));
		lines.push_back(Line(x+w, y, x+w, y-h, DirType::RIGHT));
		lines.push_back(Line(x, y-h, x+w, y-h, DirType::BOT));
		lines.push_back(Line(x, y, x, y-h, DirType::LEFT));*/

		switch(type) {
			case ChunkType::TOP_LEFT:
				lines.push_back(Line(x, y, x+w, y, DirType::TOP));
				lines.push_back(Line(x, y, x, y-h, DirType::LEFT));
				break;
			case ChunkType::TOP_MID:
				lines.push_back(Line(x, y, x+w, y, DirType::TOP));
				break;
			case ChunkType::TOP_RIGHT:
				lines.push_back(Line(x, y, x + w, y, DirType::TOP));
				lines.push_back(Line(x+w, y, x+w, y-h, DirType::RIGHT));
				break;
			case ChunkType::MID_LEFT:
				lines.push_back(Line(x, y, x, y-h, DirType::LEFT));
				break;
			//case ChunkType::MID:
				//break;
			case ChunkType::MID_RIGHT:
				lines.push_back(Line(x+w, y, x+w, y-h, DirType::RIGHT));
				break;
			case ChunkType::BOT_LEFT:
				lines.push_back(Line(x, y, x, y-h, DirType::LEFT));
				lines.push_back(Line(x, y-h, x+w, y-h, DirType::BOT));
				break;
			case ChunkType::BOT_MID:
				lines.push_back(Line(x, y-h, x+w, y-h, DirType::BOT));
				break;
			case ChunkType::BOT_RIGHT:
				lines.push_back(Line(x + w, y, x+w, y-h, DirType::RIGHT));
				lines.push_back(Line(x, y-h, x+w, y-h, DirType::BOT));
				break;
		}
	}
};

#endif // RECT_H
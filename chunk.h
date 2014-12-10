#ifndef CHUNK_H_
#define CHUNK_H_

#include "vec.h"
#include "rect.h"

enum ChunkType {TOP_LEFT, TOP_MID, TOP_RIGHT, MID_LEFT, MID, MID_RIGHT, BOT_LEFT, BOT_MID, BOT_RIGHT};

class Chunk {
public:
	Rect rect;
	Vec3 INIT_COLOR, color;
	vector<Line> lines;

	Chunk() {}

	Chunk(float x, float y, float w, float h, Vec3 color, ChunkType type)
		: rect(Rect(x, y, w, h)), INIT_COLOR(color), color(color)
	{
		switch(type) {
			case ChunkType::TOP_LEFT:
				lines.push_back(Line(x, y, x+w, y, LineType::TOP));
				lines.push_back(Line(x, y, x, y-h, LineType::LEFT));
				break;
			case ChunkType::TOP_MID:
				lines.push_back(Line(x, y, x+w, y, LineType::TOP));
				break;
			case ChunkType::TOP_RIGHT:
				lines.push_back(Line(x, y, x + w, y, LineType::TOP));
				lines.push_back(Line(x+w, y, x+w, y-h, LineType::RIGHT));
				break;
			case ChunkType::MID_LEFT:
				lines.push_back(Line(x, y, x, y-h, LineType::LEFT));
				break;
			//case ChunkType::MID:
				//break;
			case ChunkType::MID_RIGHT:
				lines.push_back(Line(x+w, y, x+w, y-h, LineType::RIGHT));
				break;
			case ChunkType::BOT_LEFT:
				lines.push_back(Line(x, y, x, y-h, LineType::LEFT));
				lines.push_back(Line(x, y-h, x+w, y-h, LineType::BOT));
				break;
			case ChunkType::BOT_MID:
				lines.push_back(Line(x, y-h, x+w, y-h, LineType::BOT));
				break;
			case ChunkType::BOT_RIGHT:
				lines.push_back(Line(x + w, y, x+w, y-h, LineType::RIGHT));
				lines.push_back(Line(x, y-h, x+w, y-h, LineType::BOT));
				break;
		}
	}
};

#endif // RECT_H
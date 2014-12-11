#ifndef PLAYER_H_
#define PLAYER_H_

#include "references.cu"
#include <vector>
#include <iostream>
#include <iomanip>

#include "rect.h"
#include "chunk.h"
#include "input.h"
#include "utils.h"


/*
 * The user controllable character of the game.
 */
class Player {
private:
	// TODO: make const
	float DT;		// player movement
	int CHUNKS;		// must be a perfect square
	int CHUNKS_PER_AXIS;

	float COLOR_INTENSITY_FALLOFF;

	ChunkType getChunkType(const int& chunkNum) const {
		if(chunkNum == 0) {
			return ChunkType::TOP_LEFT;
		}

		if(chunkNum == CHUNKS-1) {
			return ChunkType::BOT_RIGHT;
		}

		if(chunkNum == CHUNKS_PER_AXIS-1) {
			return ChunkType::TOP_RIGHT;
		}

		if(chunkNum < CHUNKS_PER_AXIS-1) {
			return ChunkType::TOP_MID;
		}

		if(chunkNum == CHUNKS-CHUNKS_PER_AXIS) {
			return ChunkType::BOT_LEFT;
		}

		if(chunkNum > CHUNKS-CHUNKS_PER_AXIS) {
			return ChunkType::BOT_MID;
		}

		if(chunkNum % CHUNKS_PER_AXIS == 0) {
			return ChunkType::MID_LEFT;
		}

		if((chunkNum+1) % CHUNKS_PER_AXIS == 0) {
			return ChunkType::MID_RIGHT;
		}

		return ChunkType::MID;
	}

public:
	Vec2 pos;
	vector<Chunk> body;		// relative to pos
	
	Player() {}

	Player(float x, float y, float w, float h)
		: pos(Vec2(x, y))
	{
		DT = 0.0015;
		CHUNKS = pow(14, 2);
		CHUNKS_PER_AXIS = sqrt(CHUNKS);
		COLOR_INTENSITY_FALLOFF = 0.05;

		// Initialize body chunks
		Vec3 color(0.1, 0.1, 0.1);
		float cx = 0;
		float cy = 0;
		const float cw = w/CHUNKS_PER_AXIS;
		const float ch = h/CHUNKS_PER_AXIS;

		for(int i=0; i < CHUNKS; ++i) {	
			body.push_back(Chunk(cx, cy, cw, ch, color, getChunkType(i)));

			if((i+1) % CHUNKS_PER_AXIS == 0 && i != 0) {
				cx = 0;
				cy -= ch;
			}
			else {
				cx += w/CHUNKS_PER_AXIS;
			}
		}
	}

	// Draws a rectangle in chunks
	void draw() const {
		for(const auto& part : body) {
			const float x = pos.x + part.rect.pos.x;
			const float y = pos.y + part.rect.pos.y;

			glColor3f(part.color.x, part.color.y, part.color.z);
			glBegin(GL_POLYGON);
				glVertex2f(x, y);
				glVertex2f(x + part.rect.size.x, y);
				glVertex2f(x + part.rect.size.x, y - part.rect.size.y);
				glVertex2f(x, y - part.rect.size.y);
			glEnd();
		}
	}

	void updatePos() {
		if(keysDown['w']) {
			pos.y += DT;
		}
		if(keysDown['a']) {
			pos.x -= DT;
		}
		if(keysDown['s']) {
			pos.y -= DT;
		}
		if(keysDown['d']) {
			pos.x += DT;
		}
		//if(input.keysDown[32]) {			// spacebar
		//	debugRays = !debugRays;
		//}
		if(keysDown[27]) {			// escape
			exit(0);
		}

		glutPostRedisplay();
	}

	// recurses through chunks
	void updateChunkColor(const int& chunkNum, DirType fromSide, const float& colorIntensity, const int& numPasses) {
		if(colorIntensity - COLOR_INTENSITY_FALLOFF <= 0) {
			return;
		}
		
		// update self
		body[chunkNum].color.add(colorIntensity);

		// update surrounding chunks
		// recurse on surrounding chunks
		vector<DirType> passedFrom;
		
		switch(body[chunkNum].type) {
			case ChunkType::TOP_LEFT:
				switch(fromSide) {
					case DirType::RIGHT:
						passedFrom.push_back(DirType::TOP);		// pass down a square
						break;
					case DirType::BOT:
						passedFrom.push_back(DirType::LEFT);	// pass right a square
						break;
					default:
						passedFrom.push_back(DirType::TOP);
						passedFrom.push_back(DirType::LEFT);
						break;
				}
				break;
			case ChunkType::TOP_MID:
				switch(fromSide) {
					case DirType::LEFT:
						passedFrom.push_back(DirType::LEFT);
						passedFrom.push_back(DirType::TOP);
						break;
					case DirType::RIGHT:
						passedFrom.push_back(DirType::RIGHT);
						passedFrom.push_back(DirType::TOP);
						break;
					case DirType::BOT:
						passedFrom.push_back(DirType::LEFT);
						passedFrom.push_back(DirType::RIGHT);
						break;
					default:
						passedFrom.push_back(DirType::LEFT);
						passedFrom.push_back(DirType::RIGHT);
						passedFrom.push_back(DirType::TOP);
						break;
				}
				break;
			case ChunkType::TOP_RIGHT:
				switch(fromSide) {
					case DirType::LEFT:
						passedFrom.push_back(DirType::TOP);
						break;
					case DirType::BOT:
						passedFrom.push_back(DirType::RIGHT);
						break;
					default:
						passedFrom.push_back(DirType::RIGHT);
						passedFrom.push_back(DirType::TOP);
						break;
				}
				break;
			case ChunkType::MID_LEFT:
				switch(fromSide) {
					case DirType::TOP:
						passedFrom.push_back(DirType::LEFT);
						passedFrom.push_back(DirType::TOP);
						break;
					case DirType::RIGHT:
						passedFrom.push_back(DirType::TOP);
						passedFrom.push_back(DirType::BOT);
						break;
					case DirType::BOT:
						passedFrom.push_back(DirType::BOT);
						passedFrom.push_back(DirType::LEFT);
						break;
					default:
						passedFrom.push_back(DirType::TOP);
						passedFrom.push_back(DirType::BOT);
						passedFrom.push_back(DirType::LEFT);
						break;
				}
				break;
			case ChunkType::MID:
				switch(fromSide) {
					case DirType::TOP:
						passedFrom.push_back(DirType::TOP);
						passedFrom.push_back(DirType::RIGHT);
						passedFrom.push_back(DirType::LEFT);
						break;
					case DirType::RIGHT:
						passedFrom.push_back(DirType::BOT);
						passedFrom.push_back(DirType::TOP);
						passedFrom.push_back(DirType::RIGHT);
						break;
					case DirType::BOT:
						passedFrom.push_back(DirType::BOT);
						passedFrom.push_back(DirType::RIGHT);
						passedFrom.push_back(DirType::LEFT);
						break;
					case DirType::LEFT:
						passedFrom.push_back(DirType::BOT);
						passedFrom.push_back(DirType::TOP);
						passedFrom.push_back(DirType::LEFT);
						break;
					default: // should never happen
						passedFrom.push_back(DirType::TOP);
						passedFrom.push_back(DirType::RIGHT);
						passedFrom.push_back(DirType::BOT);
						passedFrom.push_back(DirType::LEFT);
						break;
				}
				break;
			case ChunkType::MID_RIGHT:
				if(fromSide != DirType::TOP)	passedFrom.push_back(DirType::TOP);
				if(fromSide != DirType::RIGHT)	passedFrom.push_back(DirType::RIGHT);
				if(fromSide != DirType::BOT)	passedFrom.push_back(DirType::BOT);
				break;
			case ChunkType::BOT_LEFT:
				passedFrom.push_back(DirType::BOT);
				passedFrom.push_back(DirType::LEFT);
				break;
			case ChunkType::BOT_MID:
				passedFrom.push_back(DirType::LEFT);
				passedFrom.push_back(DirType::BOT);
				passedFrom.push_back(DirType::RIGHT);
				break;
			case ChunkType::BOT_RIGHT:
				passedFrom.push_back(DirType::RIGHT);
				passedFrom.push_back(DirType::BOT);
				break;
		}


		for(const auto& dir : passedFrom) {
			int num;
			switch(dir) {
				case DirType::TOP:
					num = chunkNum + CHUNKS_PER_AXIS;
					break;
				case DirType::RIGHT:
					num = chunkNum-1;
					break;
				case DirType::BOT:
					num = chunkNum - CHUNKS_PER_AXIS;
					break;
				case DirType::LEFT:
					num = chunkNum+1;
					break;
			}

			body[num].color.add(colorIntensity-COLOR_INTENSITY_FALLOFF);

			// TODO: limit numPasses by CHUNKS_PER_AXIS
			updateChunkColor(num, dir, colorIntensity-COLOR_INTENSITY_FALLOFF, numPasses+1);
		}
	}
} player;

#endif // PLAYER_H
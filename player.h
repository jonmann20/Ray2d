#ifndef PLAYER_H_
#define PLAYER_H_

#include "references.cu"
#include <vector>
#include <queue>
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
		CHUNKS = pow(25, 2);
		CHUNKS_PER_AXIS = sqrt(CHUNKS);
		COLOR_INTENSITY_FALLOFF = 1.1;

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
		//	game.debugRays = !game.debugRays;
		//}
		if(keysDown[27]) {			// escape
			exit(0);
		}

		glutPostRedisplay();
	}

	void updateChunkColors(const int& chunkNum, float initIntensity) {		// TODO: color blending, not just intensity
		vector<bool> setChunks(body.size(), 0);	// seperate for each ray
		queue<pair<int, float>> q({make_pair(chunkNum, initIntensity)});

		while(q.size() > 0) {
			const int chunkNum = q.front().first;
			float intensity = q.front().second;
			q.pop();

			if(!setChunks[chunkNum]) {
				//printV2(chunkNum, intensity);

				// Update current chunk
				body[chunkNum].color.add(intensity);		// linear falloff
				setChunks[chunkNum] = true;

				// Determine surrounding chunks
				intensity /= COLOR_INTENSITY_FALLOFF;


				if(intensity > 0) {
					getSurroundingChunkNums(q, intensity, chunkNum, setChunks);
				}

				//cin.get();
			}
		}
	}

	void getSurroundingChunkNums(queue<pair<int, float>>& q, const float& intensity, const int& chunkNum, vector<bool>& setChunks) {
		const pair<int, float> T = make_pair(chunkNum - CHUNKS_PER_AXIS, intensity);
		const pair<int, float> TR = make_pair(T.first + 1, intensity);
		const pair<int, float> R = make_pair(chunkNum + 1, intensity);
		const pair<int, float> BR = make_pair(chunkNum + CHUNKS_PER_AXIS + 1, intensity);
		const pair<int, float> B = make_pair(BR.first - 1, intensity);
		const pair<int, float> BL = make_pair(B.first - 1, intensity);
		const pair<int, float> L = make_pair(chunkNum - 1, intensity);
		const pair<int, float> TL = make_pair(T.first - 1, intensity);

		// Any bordering chunk (including diagonals except mid)
		switch(body[chunkNum].type) {
			case ChunkType::TOP_LEFT:	return q.push(R), q.push(BR), q.push(B);
			case ChunkType::TOP_MID:	return q.push(R), q.push(BR), q.push(B), q.push(BL), q.push(L);
			case ChunkType::TOP_RIGHT:	return q.push(B), q.push(BL), q.push(L);
			case ChunkType::MID_LEFT:	return q.push(T), q.push(TR), q.push(R), q.push(BR), q.push(B);
			case ChunkType::MID:		return q.push(T), q.push(R), q.push(B), q.push(L);
			case ChunkType::MID_RIGHT:	return q.push(T), q.push(B), q.push(BL), q.push(L), q.push(TL);
			case ChunkType::BOT_LEFT:	return q.push(T), q.push(TR), q.push(R);
			case ChunkType::BOT_MID:	return q.push(T), q.push(TR), q.push(R), q.push(L), q.push(TL);
			case ChunkType::BOT_RIGHT:	return q.push(T); q.push(L), q.push(TL);
		}
	}

} player;

#endif // PLAYER_H
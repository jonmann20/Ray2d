#ifndef PLAYER_H_
#define PLAYER_H_

#include "chunk.h"

#include <vector>
#include <queue>
using namespace std;

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

	ChunkType getChunkType(const int& chunkNum) const;

public:
	Vec2 pos;
	vector<Chunk> body;		// relative to pos
	
	Player() {}

	Player(float x, float y, float w, float h);

	void draw() const;
		// EFFECTS: Draws a rectangle in chunks

	void updatePos();
	void updateChunkColors(const int& chunkNum, float initIntensity);
	void getSurroundingChunkNums(queue<pair<int, float>>& q, const float& intensity, const int& chunkNum, vector<bool>& setChunks);
} player;

#endif // PLAYER_H
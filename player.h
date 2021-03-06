#ifndef PLAYER_H_
#define PLAYER_H_

#include "chunk.h"
#include "rect.h"

#include <vector>
#include <queue>
using namespace std;

/*
 * The user controllable character of the game.
 */
class Player {
private:
	const Vec2 size;
	const float COLOR_INTENSITY_FALLOFF;
	
	const int CHUNKS_PER_AXIS;
	int CHUNKS;		// must be a perfect square

	ChunkType getChunkType(const int& chunkNum) const;

	// updateChunkColors helpers
	void byTopL(const int& chunkNum, const float& initIntensity);
	void byTopM(const int& chunkNum, const float& initIntensity);
	void byTopR(const int& chunkNum, const float& initIntensity);
	void byMidR(const int& chunkNum, const float& initIntensity);
	void byMidL(const int& chunkNum, const float& initIntensity);
	void byBotL(const int& chunkNum, const float& initIntensity);
	void byBotM(const int& chunkNum, const float& initIntensity);
	void byBotR(const int& chunkNum, const float& initIntensity);

public:
	Vec2 pos;
	vector<Chunk> body;		// relative to pos
	
	Player();

	void draw() const;
		// EFFECTS: Draws a rectangle in chunks

	void updatePos();
	void updateChunkColors(const int& chunkNum, const float& initIntensity);

	Rect getBoundingRect() const;
};

extern Player player;

#endif // PLAYER_H
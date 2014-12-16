#include "player.h"

#include "game.h"
#include "input.h"
#include "chunk.h"
#include "light.h"
#include "color.h"
#include "utils.h"
#include "profiler.h"

#include <omp.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <vector>
#include <queue>
#include <iostream>

Player player = Player();

Player::Player()
	: DT(0.0015), COLOR_INTENSITY_FALLOFF(0.001), CHUNKS_PER_AXIS(300)
{
	CHUNKS = pow(CHUNKS_PER_AXIS, 2);

	pos = Vec2(0, 0);

	// Initialize body chunks
	const float w = 0.25;
	const float h = 0.25;
	float cx = 0;
	float cy = 0;
	const float cw = w/CHUNKS_PER_AXIS;
	const float ch = h/CHUNKS_PER_AXIS;

	for(int i=0; i < CHUNKS; ++i) {
		body.push_back(Chunk(cx, cy, cw, ch, Color::GRAY, getChunkType(i)));

		if((i+1) % CHUNKS_PER_AXIS == 0 && i != 0) {
			cx = 0;
			cy -= ch;
		}
		else {
			cx += w/CHUNKS_PER_AXIS;
		}
	}
}

ChunkType Player::getChunkType(const int& chunkNum) const {
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

void Player::draw() const {
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

void Player::updatePos() {
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

	if(keysDown[32]) {			// spacebar
		for(auto& i : game.lights) {
			i.raysVisible = !i.raysVisible;
		}
	}

	if(keysDown['p']) {
		keysDown['p'] = false;
		profiler.avg("ray collision");
	}
	// escape
	//exit(0);
	
	glutPostRedisplay();
}

void Player::byTopL(const int& chunkNum, const float& initIntensity) {
	//#pragma omp parallel for
	for(int i=0; i < CHUNKS_PER_AXIS; ++i) {
		const float newIntensity = initIntensity - i*COLOR_INTENSITY_FALLOFF;

		if(newIntensity > 0) {
			// right column - bottom right
			for(int j=0; j < i; ++j) {
				player.body[chunkNum + i + j*CHUNKS_PER_AXIS].color.add(newIntensity);
			}

			// bottom row
			for(int j=0; j <= i; ++j) {
				player.body[i*CHUNKS_PER_AXIS + j].color.add(newIntensity);
			}
		}
	}
}

void Player::byTopM(const int& chunkNum, const float& initIntensity) {
	const int offsetL = chunkNum;
	const int offsetR = CHUNKS_PER_AXIS - 1 - chunkNum;

	//#pragma omp parallel for
	for(int i=0; i < CHUNKS_PER_AXIS; ++i) {
		const float newIntensity = initIntensity - i*COLOR_INTENSITY_FALLOFF;

		if(newIntensity > 0) {
			// left column - bottom left
			if(i <= offsetL) {
				for(int j=0; j < i; ++j) {
					player.body[chunkNum - i + j*CHUNKS_PER_AXIS].color.add(newIntensity);
				}
			}

			// right column - bottom right
			if(i <= offsetR) {
				for(int j=0; j < i; ++j) {
					player.body[chunkNum + i + j*CHUNKS_PER_AXIS].color.add(newIntensity);
				}
			}

			// bottom row
			for(int j=0; j <= i; ++j) {
				// left half
				if(j <= offsetL) {
					player.body[chunkNum + i*CHUNKS_PER_AXIS - j].color.add(newIntensity);
				}
				
				// right half - middle
				if(j <= offsetR) {
					if((chunkNum + i*CHUNKS_PER_AXIS - j) != (chunkNum + i*CHUNKS_PER_AXIS + j)) {
						player.body[chunkNum + i*CHUNKS_PER_AXIS + j].color.add(newIntensity);
					}
				}
			}
		}
	}
}

void Player::byTopR(const int& chunkNum, const float& initIntensity) {
	//#pragma omp parallel for
	for(int i=0; i < CHUNKS_PER_AXIS; ++i) {
		const float newIntensity = initIntensity - i*COLOR_INTENSITY_FALLOFF;

		if(newIntensity > 0) {
			// left column - bottom left
			for(int j=0; j < i; ++j) {
				player.body[chunkNum - i + j*CHUNKS_PER_AXIS].color.add(newIntensity);
			}

			// bottom row
			for(int j=0; j <= i; ++j) {
				player.body[chunkNum + i*CHUNKS_PER_AXIS - j].color.add(newIntensity);
			}
		}
	}
}

void Player::byMidR(const int& chunkNum, const float& initIntensity) {
	const int offsetT = ((chunkNum+1) / CHUNKS_PER_AXIS) - 1;
	const int offsetB = CHUNKS_PER_AXIS - offsetT - 1;

	//#pragma omp parallel for
	for(int i=0; i < CHUNKS_PER_AXIS; ++i) {
		const float newIntensity = initIntensity - i*COLOR_INTENSITY_FALLOFF;

		if(newIntensity > 0) {
			// left column
			for(int j=0; j < i; ++j) {
				// top half
				if(j <= offsetT) {
					player.body[chunkNum - i - j*CHUNKS_PER_AXIS].color.add(newIntensity);
				}
				
				// bottom half
				if(j <= offsetB) {
					if((chunkNum - i - j*CHUNKS_PER_AXIS) != (chunkNum - i + j*CHUNKS_PER_AXIS)) {
						player.body[chunkNum - i + j*CHUNKS_PER_AXIS].color.add(newIntensity);
					}
				}
			}

			// top row
			if(i <= offsetT) {
				for(int j=0; j <= i; ++j) {
					player.body[chunkNum - i*CHUNKS_PER_AXIS - j].color.add(newIntensity);
				}
			}

			// bottom row
			if(i <= offsetB) {
				for(int j=0; j <= i; ++j) {
					player.body[chunkNum + i*CHUNKS_PER_AXIS - j].color.add(newIntensity);
				}
			}
		}
	}
}

void Player::byMidL(const int& chunkNum, const float& initIntensity) {
	const int offsetT = chunkNum / CHUNKS_PER_AXIS;
	const int offsetB = CHUNKS_PER_AXIS - offsetT - 1;

	//#pragma omp parallel for
	for(int i=0; i < CHUNKS_PER_AXIS; ++i) {
		const float newIntensity = initIntensity - i*COLOR_INTENSITY_FALLOFF;

		if(newIntensity > 0) {
			// left column
			for(int j=0; j < i; ++j) {
				// top half
				if(j <= offsetT) {
					player.body[chunkNum + i - j*CHUNKS_PER_AXIS].color.add(newIntensity);
				}

				// bottom half
				if(j <= offsetB) {
					if((chunkNum + i - j*CHUNKS_PER_AXIS) != (chunkNum + i + j*CHUNKS_PER_AXIS)) {
						player.body[chunkNum + i + j*CHUNKS_PER_AXIS].color.add(newIntensity);
					}
				}
			}

			// top row
			if(i <= offsetT) {
				for(int j=0; j <= i; ++j) {
					player.body[chunkNum - i*CHUNKS_PER_AXIS + j].color.add(newIntensity);
				}
			}

			// bottom row
			if(i <= offsetB) {
				for(int j=0; j <= i; ++j) {
					player.body[chunkNum + i*CHUNKS_PER_AXIS + j].color.add(newIntensity);
				}
			}
		}
	}
}

void Player::updateChunkColors(const int& chunkNum, const float& initIntensity) {		// TODO: color blending, not just intensity
	switch(player.body[chunkNum].type) {
		case ChunkType::TOP_LEFT:	return byTopL(chunkNum, initIntensity);
		case ChunkType::TOP_MID:	return byTopM(chunkNum, initIntensity);
		case ChunkType::TOP_RIGHT:	return byTopR(chunkNum, initIntensity);
		case ChunkType::MID_RIGHT:	return byMidR(chunkNum, initIntensity);
		case ChunkType::MID_LEFT:	return byMidL(chunkNum, initIntensity);
	}
}
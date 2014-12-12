#ifndef INPUT_H_
#define INPUT_H_

extern bool keysDown[];

void keydown(unsigned char key, int x, int y);
void keyup(unsigned char key, int x, int y);

#endif // INPUT_H
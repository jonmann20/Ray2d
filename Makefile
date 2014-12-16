src=main.cpp input.cpp vec.cpp game.cpp light.cpp color.cpp player.cpp line.cpp collision.cpp chunk.cpp rect.cpp utils.h
exe=FlappyRay.exe

#----- Default
build:
	g++ $(src) -std=c++11 -O3 -o $(exe) -lGL -lstdc++ -lc -lm -lglut -lGLU

run:
	./$(exe)

#----- Utils
clean:
	rm -f *.o *.exe

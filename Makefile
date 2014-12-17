src=main.cpp vec.cpp light.cpp player.cpp line.cpp collision.cpp chunk.cpp rect.cpp utils.h
exe=FlappyRay.exe

#----- Default
build:
	g++ $(src) -fopenmp -std=c++11 -O3 -o $(exe)

run:
	./$(exe)

#----- Utils
clean:
	rm -f *.o *.exe

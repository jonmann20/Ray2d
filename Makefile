src=main.cpp vec.cpp light.cpp player.cpp line.cpp collision.cpp chunk.cpp rect.cpp utils.h

#----- Default
build:
	g++ -fopenmp -std=c++11 -O3 $(src) -o FlappyRay.exe

iblight:
	icpc -openmp -std=c++11 -O3 $(src) -o iFlappyRay.exe

#----- Utils
clean:
	rm -f *.o *.exe

#include "light.h"
#include "vec.h"

#include <omp.h>

#include <vector>

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>

#include <iomanip>

#include <iostream>
using namespace std;


int main(int argc, char* argv[]) {
	Light light = Light(-0.2, 0.85, LightType::FLOURESCENT, Vec3(1, 1, 1), true);

	const int NUM_SAMPLES = 500;
	vector<long> times;
    
    for(int i=0; i < NUM_SAMPLES; ++i) {
		struct timeval start, end;
    	long mtime, seconds, useconds;    

    	gettimeofday(&start, NULL);

    	// profiling code
		light.checkRays();

	    gettimeofday(&end, NULL);

	    seconds  = end.tv_sec  - start.tv_sec;
	    useconds = end.tv_usec - start.tv_usec;
    	mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    	times.push_back(mtime);
	}


	// calculate stats
	long sum = 0;
	for(auto& i : times) {
		sum += i;
	}

	//cout.precision(1);
	//cout << fixed;
	cout << endl;

	cout << NUM_SAMPLES << " samples taken\n\n";
	cout << setw(9) << "min: " << setw(4) << *min_element(times.begin(), times.end()) << " ms\n";
	cout << setw(9) << "max: " << setw(4) << *max_element(times.begin(), times.end()) << " ms\n";
	cout << setw(9) << "average: " << setw(4) << sum / times.size() << " ms\n\n";


	return EXIT_SUCCESS;
}
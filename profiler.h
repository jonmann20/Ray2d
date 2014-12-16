#ifndef PROFILER_H_
#define PROFILER_H_

#include <string>
#include <vector>
using namespace std;

class Profiler {
	double PCFreq;
	__int64 CounterStart;
	vector<double> pTimes;

public:

	Profiler();

	void start();
	void end(const string& str);
	void avg(const string& str);
	void stats();
};

extern Profiler profiler;

#endif // PROFILER_H
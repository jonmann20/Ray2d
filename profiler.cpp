#include "profiler.h"

#include <iomanip>
#include <iostream>
#include <windows.h>
#include <algorithm>
#include <string>
using namespace std;

Profiler profiler = Profiler();

Profiler::Profiler()
	: PCFreq(0.0), CounterStart(0)
{}

void Profiler::start() {
	LARGE_INTEGER li;
	if(!QueryPerformanceFrequency(&li))
		cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart)/1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

void Profiler::end(const string& str) {
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);

	double t = double(li.QuadPart-CounterStart)/PCFreq;
	pTimes.push_back(t);
	cout << str << ": " << t << " ms\n";
}

void Profiler::endAvg(const string& str) {
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);

	double t = double(li.QuadPart-CounterStart)/PCFreq;
	pTimes.push_back(t);
	
	double sum = 0.0;
	for(const auto& i : pTimes) {
		sum += i;
	}

	cout.precision(3);
	cout << fixed;
	cout << str << " (avg): " << sum / pTimes.size() << " ms\n";
	pTimes.clear();
}

void Profiler::stats() {
	double sum = 0.0;

	for(const auto& i : pTimes) {
		sum += i;
	}

	cout.precision(1);
	cout << fixed;
	cout << endl;

	cout << setw(9) << "min: " << *min_element(pTimes.begin(), pTimes.end()) << " ms\n";
	cout << setw(9) << "max: " << *max_element(pTimes.begin(), pTimes.end()) << " ms\n";
	cout << setw(9) << "average: " << sum / pTimes.size() << " ms\n";
}
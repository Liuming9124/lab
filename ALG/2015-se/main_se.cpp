#include "./se.h"
#include <sstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]){
    int Run, Func, Evals, Bits, Searchers, Regions, Samples, Players;

    stringstream ss;
    for (int i = 1; i < argc; i++)
        ss << argv[i] << " ";
    ss >> Run;
    ss >> Func;
    ss >> Evals;
    ss >> Bits;
    ss >> Searchers;
    ss >> Regions;
    ss >> Samples;
    ss >> Players;

    SE se;
    se.RunALG(Run, Func, Evals, Bits, Searchers, Regions, Samples, Players);
    return 0;
}
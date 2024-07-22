#ifndef DE_H
#define DE_H


#include <queue>
#include <vector>
#include <algorithm>
#include "./problem.cpp"
#include "../AlgPrint.h"
#include "../Tool.h"
using namespace std;


class De: Problem {
public:

    typedef struct Particle{
        vector<double> _position;
        double _fitness;
    } _Particle;
    
    static bool compareFitness(const _Particle &, const _Particle &);

    void RunALG( int, int, int, int, double, double, double);

private:
    int _Pop;
    int _Run;
    int _Iter;
    int _Dim;
    double _Bounder;
    double _Cr;
    double _F;

    _Particle _Offspring;
    vector<_Particle> _Swarm;

    void Init();
    void Evaluation();
    void Reset();
    
    void CheckBorder(_Particle);

    Problem problem;
    AlgPrint show;
    Tool tool;
};

void De::RunALG(int Pop, int Run, int Iter, int Dim, double Bounder, double Cr, double F){

    _Pop = Pop;
    _Run = Run;
    _Iter = Iter;
    _Dim = Dim;
    _Bounder = Bounder;
    _Cr = Cr;
    _F = F;
    
    show = AlgPrint(_Run, "./result", "de");
    show.NewShowDataDouble(_Iter);

    problem.setStrategy(make_unique<Func1>());
    while (_Run--){
        cout << "-------------------Run" << Run - _Run << "---------------------" << endl;
        Init();
        Evaluation();
        Reset();
    }
    show.PrintToFileDouble("./result/result.txt", _Iter);
    cout << "end" << endl;
}

void De::Init(){
    _Swarm.resize(_Pop);
    _Offspring._position.resize(_Dim);
    int dim = _Dim;
    for (int i=0; i<_Pop; i++){
        _Swarm[i]._position.resize(dim);
        for (int j=0; j<dim; j++){
            _Swarm[i]._position[j] = tool.rand_double(-1 * _Bounder, _Bounder);
        }
        _Swarm[i]._fitness = problem.executeStrategy(_Swarm[i]._position, _Dim);
    }
}

void De::Evaluation(){
    for (int iter=0; iter<_Iter; iter++) {
        for (int i=0; i<_Pop; i++){
            // get random three place to mutaion
            int a, b, c;
            do {
                a = tool.rand_int(0, _Pop-1);
            } while (a==i);
            do {
                b = tool.rand_int(0, _Pop-1);
            } while( b==i || b==a);
            do {
                c = tool.rand_int(0, _Pop-1);
            } while ( c==b || c==a || c==i);

            // mutation & check boundary & crossover
            int Jrand = tool.rand_int(0, _Dim-1);
            for (int j=0; j<_Dim; j++){
                if ( tool.rand_double(0,1) < _Cr || j == Jrand){
                    _Offspring._position[i] =  _Swarm[c]._position[j] + _F * ( _Swarm[a]._position[j] - _Swarm[b]._position[j]);
                    CheckBorder(_Offspring);
                }
                else {
                    _Offspring = _Swarm[i];
                }  
            }
            // selection
            _Offspring._fitness = problem.executeStrategy(_Offspring._position, _Dim);
            if (_Offspring._fitness < _Swarm[i]._fitness){
                _Swarm[i] = _Offspring;
            }
        }
        // show data
        double best = _Swarm[0]._fitness;
        for (int p=1; p<_Pop; p++){
            if (best>_Swarm[p]._fitness)
                best = _Swarm[p]._fitness;
        }
        show.SetDataDouble(_Run, best, iter);
    }
}


void De::Reset(){
    _Offspring._fitness = 0;
}

void De::CheckBorder(_Particle check){
    for (int i = 0; i<_Dim; i++){
        if (check._position[i] < -1 * _Bounder || check._position[i] > _Bounder)
        {
            check._position[i] = tool.rand_double(-_Bounder, _Bounder);
        }
    }
}

bool De::compareFitness(const _Particle &a, const _Particle &b)
{
    return a._fitness < b._fitness;
}

#endif
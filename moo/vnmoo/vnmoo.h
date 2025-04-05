#ifndef VNMOO_H
#define VNMOO_H

#include "../Tool.h"
#include "./problem.cpp"
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

class VNMOO
{
public:
    typedef struct history
    {
        vector<double> memory; // memory[0]: CR, memory[1]: F
    } T_History;

    typedef struct point
    {
        int _index;
        double _inCR, _inF, crowdingDistance;
        vector<double> _fitness;
        vector<double> _position;
        int _rank; // Rank
        int net_best; // 儲存區域中適應值表現最好的點的索引
        double _pfitness; // (3.3) F(xi) = δ(Rmax − R(x)) + C(x),
    } T_Point;

    typedef struct net
    {
        int net_index;
        double _Ia, _Ib;
        vector<int> point_index;
        vector<double> _EVbest;
        double _EV;
        int net_best;
    } T_Net;

    typedef struct saver {
        vector<vector<double>> position;
        vector<vector<double>> fitness;
        vector<double> obj;
        vector<int> good_saver;
    } T_Saver;


public:
    void RunALG(int, int, string, int, double);

private:
    // env
    int num_Run;
    int num_Fess;
    int num_Dim;
    int num_Result;
    int num_Evals;
    int num_Netlen;
    int num_History;

    // var
    int history_index;
    vector<T_History> HistoryTable;   // history table
    vector<T_Net> Net;    // net
    vector<T_Point> X, X_previous;    // point
    T_Saver saver; // store history best
    vector<T_Point> arch;
    vector<T_Point> external_archive; // external archive

    int num_Point;
    int num_Region;

    vector<double> SF, SCR, deltaF;
    double best_value;
    // evaluation count
    int eval_count; 
        
    // tool
    // AlgPrint show;
    Tool tool;
    Problem problem;

    void Init();
    void Evaluation();

    void expected_value();
    void vision_search();
    void update_HistoryTable();
    void net_update();

    void save_to_saver();
    void PrintSaver();

    void Init_T_point(vector<T_Point> &points, int size, int dim, bool posInitRand);
    void Init_T_net(vector<T_Net> &nets, int size, int num_region);

    static bool NetCompareFitness(const T_Net &a, const T_Net &b);
    int NetSelectTopPBest(vector<T_Net> xx, double p);
    static bool PointCompareFitness(const T_Point &a, const T_Point &b);

    void CrowdingDistanceAssignment(vector<int>& front, vector<T_Point>& points);
    vector<vector<int>> FastNonDominatedSort(vector<T_Point>& points);
    double MeasureFitness(T_Point& point, vector<T_Point>& allPoints, double delta);
};

void VNMOO::RunALG(int Run, int Evals, string Func, int NetLength, double History)
{
    problem.setStrategy(Func);
    num_Run = Run;
    num_Fess = Evals;
    num_Netlen = NetLength;
    num_Region = (num_Netlen-1)*(num_Netlen-1);
    num_Point = num_Netlen*num_Netlen;
    num_History = History * num_Dim;
    num_Dim = problem.getFunctionDimension();
    num_Result = problem.getResultCount();

    while (num_Run--)
    {
        cout << "-------------------Run" << Run - num_Run << "---------------------" << endl;
        Init();
        Evaluation();
    }
    // // show.PrintToFileDouble("./result/result" + Func + "_DIM" + to_string(num_Dim) + "_NetLen" + to_string(num_Netlen) + "_Fess" + to_string(num_Fess) + ".txt", num_Fess);
    // cout << "end" << endl;
}

void VNMOO::Init()
{
    // show.init();
    eval_count = 0;
    best_value = 0;

    // history memory size
    history_index = 0;
    HistoryTable.resize(num_History);
    for (int i = 0; i < num_History; i++) {
        HistoryTable[i].memory.resize(2, 0.3);
    }
    SCR.resize(num_Point);
    SF.resize(num_Point);
    // init point
    Init_T_point( X,          num_Point, num_Dim, true);
    Init_T_point( X_previous, num_Point, num_Dim, false);
    // init net
    Init_T_net(Net, num_Region, num_Netlen-1);
    
    // Initialize saver
    saver.position.clear();
    saver.fitness.clear();
    saver.obj.clear();
    saver.good_saver.assign(10, 0); // TODO checkout : Save top 10 points
}

void VNMOO::Evaluation()
{
    while (eval_count < num_Fess)
    {
        // expected value
        expected_value();
        // vision search
        vision_search();
        // net update
        net_update();

        save_to_saver();
        
    }

    PrintSaver();

    // // print the Pareto Front
    // // 先排序X，把rank=0的放到前面
    // vector<T_Point> tmp = X;
    // sort(tmp.begin(), tmp.end(), PointCompareFitness);
    // // cout << "Pareto Front" << endl;
    // // print all rank of points
    // FastNonDominatedSort(tmp);
    // vector<T_Point> pareto_front;
    // for (const auto& point : tmp) {
    //     if (point._rank == 0) {
    //         pareto_front.push_back(point);
    //     }
    // }
    // for (const auto& point : pareto_front) {
    //     for (double val : point._fitness) {
    //         cout << val << " ";
    //     }
    //     cout << endl;
    // }
}

void VNMOO::save_to_saver() {
    // 遍歷所有的點，檢查是否需要更新 saver
    for (size_t i = 0; i < X.size(); ++i) {
        double new_score = X[i]._pfitness;

        // (a) 若 saver 尚未儲存 10 筆資料，直接加入
        if (saver.position.size() < 500) {
            saver.position.push_back(X[i]._position); // 儲存位置
            saver.fitness.push_back(X[i]._fitness);  // 儲存適應值向量
            saver.obj.push_back(new_score);          // 儲存 pfitness
            saver.good_saver.push_back(X[i]._index); // 儲存索引
        }
        else {
            // (b) 若 saver 已儲存 10 筆，找出目前 `saver` 中 pfitness 最大(最差)的一筆
            // 假設越小的 pfitness 越好
            auto worst_it = std::max_element(saver.obj.begin(), saver.obj.end());
            double worst_val = *worst_it;
            int worst_idx = std::distance(saver.obj.begin(), worst_it);

            // 如果新解的 pfitness 比目前 `saver` 最差的解更好，就替換
            if (new_score < worst_val) {
                saver.position[worst_idx] = X[i]._position; // 更新位置
                saver.fitness[worst_idx] = X[i]._fitness;   // 更新適應值向量
                saver.obj[worst_idx] = new_score;           // 更新 pfitness
                saver.good_saver[worst_idx] = X[i]._index;  // 更新索引
            }
        }
    }
}

void VNMOO::PrintSaver() {
    cout << "Top 10 Solutions in Saver:" << endl;
    for (size_t i = 0; i < saver.position.size(); ++i) {
        cout << "Solution " << i + 1 << ": ";
        cout << "pfitness = " << saver.obj[i] << ", Position = [";
        for (size_t d = 0; d < saver.position[i].size(); ++d) {
            cout << saver.position[i][d];
            if (d < saver.position[i].size() - 1) cout << ", ";
        }
        cout << "]" << endl;
    }
}



void VNMOO::expected_value() {
    // cout << "Expected Value" << endl;
    const double delta = 2.0; // 額外權重 δ，用於加強非支配等級的影響
    vector<double> fitnessValues(X.size(), 0.0); // 用於存放適應值 F(x_i)
    // Step 1: 計算非支配等級和擁擠距離
    vector<T_Point> allPoints = X; // 遠見網的點
    allPoints.insert(allPoints.end(), external_archive.begin(), external_archive.end()); // 加入外部儲存器的點

    vector<vector<int>> fronts = FastNonDominatedSort(allPoints); // 進行非支配排序

    // // 印出fronts test
    // for (size_t i = 0; i < fronts.size(); i++) {
    //     cout << "Front " << i + 1 << ": ";
    //     for (int idx : fronts[i]) {
    //         cout << idx << " ";
    //     }
    //     cout << endl;
    // }
    // 計算每個 Pareto 前沿的擁擠距離和非支配等級
    for (size_t i = 0; i < fronts.size(); i++) {
        CrowdingDistanceAssignment(fronts[i], allPoints); // 為每一層計算擁擠距離
        for (int idx : fronts[i]) {
            allPoints[idx]._rank = i + 1; // 設置非支配等級(從1開始)
        }
    }

    // 找到最高的非支配等級
    int Rmax = 0;
    for (const auto& point : allPoints) {
        Rmax = max(Rmax, point._rank);
    }

    // Step 2: 計算適應值 F(x_i)
    for (size_t i = 0; i < allPoints.size(); i++) {
        fitnessValues[i] = MeasureFitness(allPoints[i], allPoints, delta); // 計算適應值
    }
    // Step 3: 計算區域期望值 ei
    vector<double> e1(num_Region, 0.0); // 搜尋頻率
    vector<double> e2(num_Region, 0.0); // 平均進步值
    vector<double> e3(num_Region, 0.0); // 平均表現值
    const double alpha = 0.7; // 進步衰退參數 α

    for (int i = 0; i < num_Region; i++) {
        double search_count = Net[i]._Ia; // 該區域被搜尋的次數
        double non_search_count = Net[i]._Ib; // 該區域未被搜尋的次數
        e1[i] = non_search_count / search_count; // 計算搜尋頻率 e1_i

        double progress_sum = 0.0; // 區域內所有點的進步值總和
        for (int idx : Net[i].point_index) {
            double progress = X_previous[idx]._pfitness - X[idx]._pfitness; // 適應值進步
            progress_sum += max(0.0, progress); // 確保進步值為正
        }
        e2[i] = alpha * e2[i] + (1.0 - alpha) * (progress_sum / Net[i].point_index.size()); // 平均進步值

        double max_fitness = numeric_limits<double>::lowest(); // 初始化為最小值
        for (int idx : Net[i].point_index) {
            max_fitness = max(max_fitness, X[idx]._pfitness); // 找到區域內的最大適應值
        }
        e3[i] = max_fitness; // 平均表現值為區域內的最佳適應值
    }
    
    // Step 4: 正規化 e1, e2, e3 並計算 ei
    double max_e1 = *max_element(e1.begin(), e1.end());
    double min_e1 = *min_element(e1.begin(), e1.end());
    double max_e2 = *max_element(e2.begin(), e2.end());
    double min_e2 = *min_element(e2.begin(), e2.end());
    double max_e3 = *max_element(e3.begin(), e3.end());
    double min_e3 = *min_element(e3.begin(), e3.end());

    for (int i = 0; i < num_Region; i++) {
        e1[i] = (max_e1 == min_e1) ? 1.0 : (e1[i] - min_e1) / (max_e1 - min_e1);
        e2[i] = (max_e2 == min_e2) ? 1.0 : (e2[i] - min_e2) / (max_e2 - min_e2);
        e3[i] = (max_e3 == min_e3) ? 1.0 : (e3[i] - min_e3) / (max_e3 - min_e3);

        Net[i]._EV = e1[i] + e2[i] + e3[i]; // 綜合計算區域期望值
    }
    // 更新前一代的點
    for (int i = 0; i < num_Point; i++) {
        X_previous[i] = X[i];
    }
    // cout << "Expected Value Done" << endl;
}

void VNMOO::vision_search() {
    // cout << "Vision Search" << endl;
    const double alpha = 0.7; // 用於進步值衰退
    const double delta = 2.0; // 額外權重，用於強化非支配等級影響
    const double pbest_ratio = 0.2; // 參考前 pbest 比例的區域 TODO add in algorithm parameters
    // 更新遠見網中每個網格點
    for (int i = 0; i < num_Point; ++i) {
        // Step 1: 隨機生成交配率 c 和步伐大小 f
        // CR
        X[i]._inCR = tool.rand_normal(HistoryTable[history_index].memory[0], 0.1);
        if (X[i]._inCR > 1)
            X[i]._inCR = 1;
        else if (X[i]._inCR < 0)
            X[i]._inCR = 0;
        // F
        X[i]._inF = 0;
        do
        {
            X[i]._inF = tool.rand_cauchy(HistoryTable[history_index].memory[1], 0.1);
            if (X[i]._inF >= 1)
                X[i]._inF = 1;
        } while (X[i]._inF <= 0);

        double c = X[i]._inCR;
        double f = X[i]._inF;
        
        // Step 2: 從前 pbest 比例的區域中隨機選擇區域 rj
        int num_top_regions = max(1, static_cast<int>(num_Region * pbest_ratio));
        if (num_top_regions < 3) num_top_regions = 3; // 至少選擇 3 個區域
        vector<T_Net> sorted_nets = Net; // 排序的區域
        sort(sorted_nets.begin(), sorted_nets.end(), NetCompareFitness);
        int selected_region_idx = sorted_nets[tool.rand_int(0, num_top_regions - 1)].net_index;

        // Step 3: 隨機選擇兩個不同的區域 rr1 和 rr2
        int rr1_idx, rr2_idx;
        
        vector<int> candidate_regions; // 候選區域
        for (int i = 0; i < num_Region; i++) {
            if (i != selected_region_idx) {
                candidate_regions.push_back(i);
            }
        }
        rr1_idx = candidate_regions[tool.rand_int(0, candidate_regions.size() - 1)];
        candidate_regions.erase(remove(candidate_regions.begin(), candidate_regions.end(), rr1_idx), candidate_regions.end());
        rr2_idx = candidate_regions[tool.rand_int(0, candidate_regions.size() - 1)];

        // Step 4: 進行交配與突變
        T_Point& parent = X[i]; // 當前網格點
        T_Point child = parent; // 子代初始化為父代
        int forced_dimension = tool.rand_int(0, num_Dim - 1); // 強制交配的一個維度

        for (int d = 0; d < num_Dim; ++d) {
            if (d == forced_dimension || tool.rand_double(0.0, 1.0) < c) {
                // 交配與突變公式
                int best_idx = Net[selected_region_idx].net_best; // rj 中表現最好的點 TODO 把net_best加入struct
                int rr1_idx_rand = Net[rr1_idx].point_index[tool.rand_int(0, 3)];
                int rr2_idx_rand = Net[rr2_idx].point_index[tool.rand_int(0, 3)];
                child._position[d] = parent._position[d]
                    + f * (X[best_idx]._position[d] - parent._position[d])
                    + f * (X[rr1_idx_rand]._position[d] - X[rr2_idx_rand]._position[d]);
            }
        }
        // update Ia, Ib
        for (int i = 0; i < num_Region; i++)
        {
            if (i == selected_region_idx)
            {
                Net[i]._Ia++;
                Net[i]._Ib = 1;
            }
            else{
                Net[i]._Ia = 1;
                Net[i]._Ib++;
            }
        }

        // Step 5: 適應值計算
        child._fitness = problem.executeStrategy(child._position);
        eval_count++;
        child._pfitness = MeasureFitness(child, X, delta); // 計算 F(xi) = δ(Rmax − R(x)) + C(x)

        // Step 6: 更新成功歷史表
        if (child._pfitness < parent._pfitness) { // 若子代優於父代
            SCR.push_back(c); // 成功交配率
            SF.push_back(f);  // 成功步伐大小
            deltaF.push_back(parent._pfitness - child._pfitness); // 成功進步值
            parent = child; // 更新父代為子代
        }
    }

    // Step 7: 更新成功歷史表的平均值 TODO checkout
    update_HistoryTable();
    // cout << "Vision Search Done" << endl;
}

void VNMOO::update_HistoryTable() {
    // cout << "Update History Table" << endl;

    // Step 1: 計算加權 w_i
    vector<double> weights(deltaF.size(), 0.0);
    double weightSum = 0.0;
    for (size_t i = 0; i < deltaF.size(); ++i) {
        weightSum += deltaF[i];
    }
    for (size_t i = 0; i < deltaF.size(); ++i) {
        weights[i] = deltaF[i] / weightSum; // w_i = ΔF_i / ΣΔF_i
    }

    // Step 2: 計算 Hc_k 和 Hf_k
    double numeratorC = 0.0, denominatorC = 0.0;
    double numeratorF = 0.0, denominatorF = 0.0;

    for (size_t i = 0; i < deltaF.size(); ++i) {
        numeratorC += weights[i] * SCR[i] * SCR[i]; // Σ(w_i * c_i^2)
        denominatorC += weights[i] * SCR[i];           // Σ(w_i * c_i)

        numeratorF += weights[i] * SF[i] * SF[i]; // Σ(w_i * f_i^2)
        denominatorF += weights[i] * SF[i];           // Σ(w_i * f_i)
    }

    if (denominatorC == 0)
        denominatorC = 1e-6;
    if (denominatorF == 0)
        denominatorF = 1e-6;

    double Hc_k = numeratorC / denominatorC; // Hc_k = Σ(w_i * c_i^2) / Σ(w_i * c_i)
    double Hf_k = numeratorF / denominatorF; // Hf_k = Σ(w_i * f_i^2) / Σ(w_i * f_i)

    // Step 3: 更新成功歷史表
    HistoryTable[history_index].memory[0] = Hc_k; // 更新交配率 Hc_k
    HistoryTable[history_index].memory[1] = Hf_k; // 更新步伐大小 Hf_k

    // 更新歷史索引（循環）
    history_index = (history_index + 1) % num_History;

    // cout << "Update History Table Done" << endl;
}


void VNMOO::net_update() {
    // cout << "Net Update" << endl;
    // 設定必要參數
    double l_min = 3.0;  // 最小邊長（依問題需要調整）
    int l_current = num_Netlen; // 當前的邊長（遠見網的邊長）
    int max_eval = num_Fess; // 最大評估次數
    double t = eval_count;   // 當前評估次數
    // 計算新的遠見網邊長 l_new
    double l_new = floor(((l_min - l_current) / max_eval) * t + l_current);
    // 更新新的 Netlen 及 num_Region 和 num_Point
    num_Netlen = (int) l_new;
    num_Region = (num_Netlen - 1) * (num_Netlen - 1);
    num_Point = num_Netlen * num_Netlen;
    // 如果邊長未改變則不需要更新
    if (l_new == l_current){
        // cout << "net update done" << endl;
        return;
    }
    // 計算新的區域數量和點數量
    int new_region_size = pow(l_new - 1, 2); // 區域數
    int new_point_size = pow(l_new, 2);      // 網格點數
    // 收集網格點的表現，基於 pfitness 排序
    vector<pair<int, double>> point_performance; // <網格點索引, pfitness 值>
    for (int i = 0; i < num_Point; i++) {
        point_performance.push_back({i, X[i]._pfitness});
    }
    // 按表現值從小到大排序（最差點優先）
    sort(point_performance.begin(), point_performance.end(), [](auto &a, auto &b) {
        return a.second < b.second;
    });
    // 保留表現較好的前 new_point_size 個網格點
    vector<T_Point> new_points;
    for (int i = 0; i < new_point_size; i++) {
        new_points.push_back(X[point_performance[i].first]);
    }
    // 更新網格點列表
    X = new_points;
    num_Point = new_point_size;
    // renumber X
    for (int i = 0; i < num_Point; i++) {
        X[i]._index = i;
    }
    // 重組遠見網
    vector<T_Net> new_net;
    Init_T_net(new_net, num_Region, num_Netlen - 1);
    Net = new_net;
    num_Region = new_region_size;

    // // 輸出日誌（可選）
    // cout << "Updated vision net: " << endl;
    // cout << "New edge length: " << l_new << endl;
    // cout << "New region size: " << new_region_size << endl;
    // cout << "New point size: " << new_point_size << endl;

    // // 印出新的遠見網（可選）
    // for (int i = 0; i < num_Region; i++) {
    //     cout << "Region " << i << ": ";
    //     for (int idx : Net[i].point_index) {
    //         cout << idx << " ";
    //     }
    //     cout << endl;
    // }

    // cout << "Net Update Done" << endl;
}


bool VNMOO::NetCompareFitness(const T_Net &a, const T_Net &b)
{
    // return descending order
    return a._EV > b._EV;
}

int VNMOO::NetSelectTopPBest(vector<T_Net> xx, double p)
{
    vector<T_Net> tmp = xx;
    sort(tmp.begin(), tmp.end(), NetCompareFitness);
    int place = tool.rand_int(0, static_cast<int>(p * num_Region));
    return tmp[place].net_index;
}

bool VNMOO::PointCompareFitness(const T_Point &a, const T_Point &b)
{
    return a._pfitness > b._pfitness;
}

void VNMOO::Init_T_point(vector<T_Point> &points, int size, int dim, bool posInitRand){
    points.resize(size);
    for (int i=0; i<size; i++){
        points[i]._index = i;
        points[i]._position.assign(dim, 0);
        points[i]._inCR = points[i]._inF = 0;
        points[i]._fitness.assign(num_Result, 0);
        points[i]._rank = 0;
        points[i].crowdingDistance = 0;
        points[i]._pfitness = 0;
        if (posInitRand){
            for (int j = 0; j < dim; j++)
                points[i]._position[j] = tool.rand_double(problem.getBounderMin()[j], problem.getBounderMax()[j]);
            points[i]._fitness = problem.executeStrategy(points[i]._position);
            eval_count++;
        }
    }
}

void VNMOO::Init_T_net(vector<T_Net> &Net, int size, int num_region){
    Net.resize(size);
    for (int i=0; i<num_region; i++){
        for (int j=0; j<num_region; j++){
            int net_idx = i * num_region + j;
            Net[net_idx].net_index = i*(num_region)+j;
            Net[net_idx].point_index.assign(4, 0);
            Net[net_idx].point_index[0] = i*num_Netlen + j;
            Net[net_idx].point_index[1] = i*num_Netlen + j + 1;
            Net[net_idx].point_index[2] = (i+1)*num_Netlen + j;
            Net[net_idx].point_index[3] = (i+1)*num_Netlen + j + 1;
            Net[net_idx]._Ia = Net[i*(num_region)+j]._Ib = 1;
            Net[net_idx]._EV = 0;

            int best_idx = Net[net_idx].point_index[0];
            double best_fitness = X[best_idx]._pfitness; // 假設適應值為最小化目標
            for (int idx : Net[net_idx].point_index) {
                // 如果新點支配當前最佳點，更新最佳點索引
                if (X[idx]._pfitness < best_fitness) {
                    best_idx = idx;
                } else if (X[idx]._pfitness == best_fitness) {
                    // 如果兩者互不支配，則比較擁擠距離，選擇擁擠距離較大的點
                    if (X[idx].crowdingDistance > X[best_idx].crowdingDistance) {
                        best_idx = idx;
                    }
                }
            }

            Net[net_idx].net_best = best_idx;
        }
    }
}

// ############################################################################################################

bool dominates(const vector<double>& a, const vector<double>& b) {
    bool at_least_one_better = false;

    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] > b[i])  // 假設目標是最小化
            return false; // a 不支配 b
        if (a[i] < b[i])
            at_least_one_better = true; // a 至少在一個目標上優於 b
    }

    return at_least_one_better;
}

vector<vector<int>> VNMOO::FastNonDominatedSort(vector<T_Point>& points) {
    // cout << "Fast Non Dominated Sort" << endl;
    vector<vector<int>> fronts; // 儲存每一層 Pareto 前沿
    fronts.emplace_back();      // 初始化第一層前沿
    int size = points.size();
    vector<int> dominationCount(size, 0);         // 每個點被支配的次數
    vector<vector<int>> dominatedSet(size);       // 每個點支配的點集合
    // 計算支配集合和支配次數
    for (int p = 0; p < size; p++) {
        for (int q = 0; q < size; q++) {
            if (p == q) continue;

            bool dominatesPtoQ = true;
            bool atLeastOneBetter = false;

            for (int k = 0; k < num_Result; k++) {
                if (points[p]._fitness[k] > points[q]._fitness[k]) {
                    dominatesPtoQ = false; // p 不支配 q
                }
                if (points[p]._fitness[k] < points[q]._fitness[k]) {
                    atLeastOneBetter = true; // p 在至少一個目標上優於 q
                }
            }

            if (dominatesPtoQ && atLeastOneBetter) {
                dominatedSet[p].push_back(q);
            } else if (!atLeastOneBetter) {
                dominationCount[p]++;
            }
        }

        if (dominationCount[p] == 0) {
            points[p]._rank = 0; // 第一層 Rank
            fronts[0].push_back(p);
        }
    }
    // 按層次分類非支配前沿
    int currentRank = 1;
    while (!fronts[currentRank - 1].empty()) {
        vector<int> nextFront;
        for (int p : fronts[currentRank - 1]) {
            for (int q : dominatedSet[p]) {
                dominationCount[q]--;
                if (dominationCount[q] == 0) {
                    points[q]._rank = currentRank + 1;
                    nextFront.push_back(q);
                }
            }
        }

        if (!nextFront.empty()) {
            fronts.push_back(nextFront);
        } else {
            break; // 無更多點可處理
        }
        currentRank++;
    }

    // cout << "Fast Non Dominated Sort Done" << endl;
    return fronts;
}

void VNMOO::CrowdingDistanceAssignment(vector<int>& front, vector<T_Point>& points) {
    int size = front.size();
    if (size == 0) return;

    for (int i : front) {
        points[i].crowdingDistance = 0.0; // 初始化擁擠距離
    }

    for (int m = 0; m < num_Result; m++) {
        sort(front.begin(), front.end(), [&points, m](int a, int b) {
            return points[a]._fitness[m] < points[b]._fitness[m];
        });

        points[front[0]].crowdingDistance = numeric_limits<double>::infinity();   // 邊界點
        points[front[size - 1]].crowdingDistance = numeric_limits<double>::infinity(); // 邊界點

        double range = points[front[size - 1]]._fitness[m] - points[front[0]]._fitness[m];
        if (range == 0) range = 1e-6; // 防止除以 0

        for (int i = 1; i < size - 1; i++) {
            points[front[i]].crowdingDistance += 
                (points[front[i + 1]]._fitness[m] - points[front[i - 1]]._fitness[m]) / range;
        }
    }
}

// (3.3) F(xi) = δ(Rmax − R(x)) + C(x),
double VNMOO::MeasureFitness(T_Point& point, vector<T_Point>& allPoints, double delta) {
    double Rmax = 0;
    for (const auto& p : allPoints) {
        if (p._rank > Rmax) {
            Rmax = p._rank;
        }
    }

    // 計算適應值 F(xi)
    double fitness = delta * (Rmax - point._rank) + point.crowdingDistance;
    return fitness;
}


#endif
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

using namespace std;

// 計算兩點之間的歐幾里得距離
double euclidean_distance(const vector<double>& a, const vector<double>& b) {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

// 讀取檔案中的 Pareto Front 或解集合
vector<vector<double>> read_points_from_file(const string& file_path, int num_objectives) {
    vector<vector<double>> points;
    ifstream file(file_path);

    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << file_path << endl;
        return points;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> point(num_objectives);
        for (int i = 0; i < num_objectives; ++i) {
            ss >> point[i];
        }
        points.push_back(point);
    }

    file.close();
    return points;
}

// 計算 IGD
double calculate_IGD(const vector<vector<double>>& pareto_front, const vector<vector<double>>& solutions) {
    double total_distance = 0.0;
    int num_pareto_points = pareto_front.size();

    // 對於每個真實 Pareto 前沿上的點
    for (const auto& pareto_point : pareto_front) {
        double min_distance = numeric_limits<double>::infinity();

        // 計算該點到解集合中最近點的距離
        for (const auto& solution : solutions) {
            double distance = euclidean_distance(pareto_point, solution);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }

        // 累加最小距離
        total_distance += min_distance;
    }

    // 返回平均距離
    return total_distance / num_pareto_points;
}

// 主函數：示範使用
int main() {
    string pareto_file = "UF1_opt.txt";  // 真實 Pareto Front 檔案
    string solution_file = "solutions.txt";  // 解集合檔案
    int num_objectives = 2;  // 目標數目（依問題設定）

    // 讀取真實 Pareto Front 和解集合
    vector<vector<double>> pareto_front = read_points_from_file(pareto_file, num_objectives);
    vector<vector<double>> solutions = read_points_from_file(solution_file, num_objectives);

    // 檢查資料是否讀取成功
    if (pareto_front.empty() || solutions.empty()) {
        cerr << "Error: Failed to load data." << endl;
        return -1;
    }

    // 計算 IGD
    double igd = calculate_IGD(pareto_front, solutions);
    cout << "IGD: " << igd << endl;

    return 0;
}

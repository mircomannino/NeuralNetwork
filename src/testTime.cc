#include <iostream>
#include <chrono>
#include "../include/neuron.h"
#include "../include/layer.h"
#include "../include/neuralNetwork.h"
#include "../include/utils.h"
using namespace std;

int main() {

    constexpr int N = 20;

    double totalTime = 0.0;
    for(int i = 0; i < N; i++) {
        auto startTime = std::chrono::steady_clock::now();
        vector<int> architecture = {5, 3, 1};

        NeuralNetwork nn(architecture, "sigmoid");

        std::vector<double> input{0.5, 0.5, 0.5, 0.5, 0.5};
        std::vector<double> output = nn.predict(input);
        // std::cout << "output: " << output[0] << std::endl;

        auto endTime = std::chrono::steady_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()/1000.0;
    }
    std::cout << "time: " << totalTime/N  << std::endl;
}

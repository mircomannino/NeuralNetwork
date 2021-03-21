#include <iostream>
#include <chrono>
#include "../include/neuron.h"
#include "../include/layer.h"
#include "../include/neuralNetwork.h"
#include "../include/utils.h"
using namespace std;

int main() {

    constexpr int N = 1;

    double totalTime = 0.0;
    for(int i = 0; i < N; i++) {
        auto startTime = std::chrono::steady_clock::now();

        vector<int> architecture = {2, 4, 8};
        std::vector<std::string> activations = {"sigmoid", "softmax"};
        NeuralNetwork nn(architecture, activations);

        nn.loadWeigths("./trainedModels/trained_model.txt");
        // nn.printInfo();
        std::vector<double> input1{2048/2048.0, 7/7.0};
        std::vector<double> input2{70/2048.0, 5/7.0};
        std::vector<double> output = nn.predict(input2);
        std::cout << "output: " << output.size() << std::endl;

        auto endTime = std::chrono::steady_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count()/1000.0;

        std::cout << "output: "; printVector(output);
        std::cout << "Prediction: " << argMax(output) << std::endl;
    }
    std::cout << "time: " << totalTime/N  << std::endl;


}

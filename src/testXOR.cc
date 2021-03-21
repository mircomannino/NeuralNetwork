#include <iostream>
#include "../include/neuron.h"
#include "../include/layer.h"
#include "../include/neuralNetwork.h"
#include "../include/utils.h"
using namespace std;

int main() {
    vector<int> architecture = {2, 3, 1};

    NeuralNetwork nn(architecture, std::vector<std::string>{"sigmoid", "sigmoid"});

    double learingRate = 0.1;
    int nEpochs = 10000;
    int miniBatchSize = 4;
    // nn.printInfo();
    nn.train("./dataXOR/XOR.train", learingRate, nEpochs, miniBatchSize);

    // Pseudo Test
    nn.test("./dataXOR/XOR.test");


    return 0;
}

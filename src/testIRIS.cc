#include <iostream>
#include "../include/neuron.h"
#include "../include/layer.h"
#include "../include/neuralNetwork.h"
#include "../include/utils.h"
using namespace std;

int main() {
    vector<int> architecture = {4, 5, 3, 3};
    cout << "SETOSA: [1, 0, 0]" << endl;
    cout << "VERSICOLOR: [0, 1 0]" << endl;
    cout << "VIRGINICA: [0, 0, 1]" << endl;

    NeuralNetwork nn(architecture, "sigmoid");

    double learingRate = 0.05;
    int nEpochs = 1000;
    int miniBatchSize = 90;
    // nn.printInfo();
    nn.train("./dataIRIS/iris.train", learingRate, nEpochs, miniBatchSize);
    // nn.printInfo();

    nn.test("./dataIRIS/iris.test");

    // Pseudo Test
    // vector<double> testSETOSA = {5.0, 3.5, 1.3, 0.3};
    // auto prediction = nn.predict(testSETOSA);
    // cout << "Real [1, 0, 0]" << " - Predicted: [" << prediction[0] << ", " << prediction[1] << ", " << prediction[2] << "]\n";

    return 0;
}

#include "../include/neuralNetwork.h"
using namespace std;

int main() {
    vector<int> architecture = {4, 3, 2, 3, 4};
    string activationType = "sigmoid";

    NeuralNetwork nn(architecture, activationType);

    double learingRate = 0.01;
    int nEpochs = 5000;
    int miniBatchSize = 250;
    nn.train("./dataAUTOENCODER/AUTOENCODER.train", learingRate, nEpochs, miniBatchSize);

    // Non si fa, mannaggia alla miseria
    nn.test("./dataAUTOENCODER/AUTOENCODER.train");

}

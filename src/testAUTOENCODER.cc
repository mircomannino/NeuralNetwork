#include "../include/neuralNetwork.h"
using namespace std;

int main() {
    vector<int> architecture = {4, 3, 3, 2, 4};
    string activationType = "sigmoid";

    NeuralNetwork nn(architecture, activationType);

    double learingRate = 0.9;
    int nEpochs = 1000;
    int miniBatchSize = 256;
    nn.train("./dataAUTOENCODER/AUTOENCODER.train", learingRate, nEpochs, miniBatchSize);

    // Non si fa, mannaggia alla miseria
    nn.test("./dataAUTOENCODER/AUTOENCODER.train");

}

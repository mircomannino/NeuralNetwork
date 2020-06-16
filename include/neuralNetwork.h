#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iomanip>
#include "layer.h"

class NeuralNetwork {
private:
    // Attributes
    std::vector<Layer> layers;
    std::vector<int> architecture;
    // Methodss
    std::vector<std::vector<double>> loadFromFile(const std::string& filePath);
    void dividePatternsAnsTargets(
        std::vector<std::vector<double>> data,
        std::vector<std::vector<double>>& targets,
        std::vector<std::vector<double>>& patterns
    );
public:
    NeuralNetwork(const std::vector<int> architecture, const std::string activationType);
    std::vector<double> predict(const std::vector<double> x);
    std::vector<double> predictAndSave(const std::vector<double> x, std::vector<std::vector<double>>& intermediateOutput);
    void train(const std::string& filePath, double learingRate, int nEpochs, int miniBatchSize);
    double test(const std::string& filePath);
    void printInfo() const;
};

#endif

#ifndef __NEURON__
#define __NEURON__

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <math.h>
#include "activationFunction.h"
#include "utils.h"

class Neuron {
private:
    double bias;
    std::vector<double> weight;
    std::vector<double> gradient;
    ActivationFunction* activationFunction;
    double activationValue;
    double deltaError;
public:
    Neuron() {}
    Neuron(const int weightDim, const std::string activationType);
    inline double getBias() const { return bias; }
    inline double getActivationValue() const { return activationValue; }
    inline ActivationFunction* getActivationFunction() const { return activationFunction; }
    inline int getWeightDim() const { return weight.size(); }
    inline std::vector<double> getWeight() const { return weight; }
    inline std::vector<double> getGradient() const {return gradient; }
    inline double getWeight(const int index) const { return weight[index]; }
    inline double getGradient(const int index) const { return gradient[index]; }
    inline double getDeltaError() const { return deltaError; }
    inline void setWeight(const int index, const double value) { this->weight[index] = value; }
    inline void setDeltaError(const double deltaError_) { deltaError = deltaError_; }
    inline void setGradient(const int index, const double value) { this->gradient[index] = value; }
    inline void setBias(const double bias_) { bias = bias_; }
    double computeOutput(const std::vector<double> x);
    void printInfo() const;
};

#endif

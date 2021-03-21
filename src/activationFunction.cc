#include "../include/activationFunction.h"

/* softmax activation */
double SoftmaxActivation::computeActivation(const std::size_t neuronIndex, std::vector<double>& all_a) {
    assert(neuronIndex < all_a.size());
    // Sum all the component
    double sum = 0.0;
    double expValNeuron = 0.0;
    for(int i = 0; i < all_a.size(); i++) {
        sum += exp(all_a[i]);
        if(i == neuronIndex) expValNeuron = exp(all_a[i]);
    }
    return expValNeuron / sum;
}

/* Linear activation */
double LinearActivation::computeActivation(const double a) {
    return a;
}
double LinearActivation::computeActivationPrime(const double a) {
    if(a >= 0) return 1.0; else -1.0;
}

/* Sigmoid activation */
double SigmoidActivation::computeActivation(const double a) {
    return (double(1.0) / double(1.0 + exp(-a)));
}
double SigmoidActivation::computeActivationPrime(const double a) {
    return (this->computeActivation(a) * (1 - this->computeActivation(a)));
}

/* ReLU activation */
double ReLUActivation::computeActivation(const double a) {
    if(a < 0) return 0;
    return a;
}
double ReLUActivation::computeActivationPrime(const double a) {
    if(a <= 0) return 0;
    return 1;
}

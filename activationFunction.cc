#include "include/activationFunction.h"

/* Sigmoid activation */
double SigmoidActivation::computeActivation(const double a) {
    return (double(1.0) / double(1.0 + exp(-a)));
}
double SigmoidActivation::computeActivationPrime(const double a) {
    return (this->computeActivation(a) * (1 - this->computeActivation(a)));
}

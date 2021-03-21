#ifndef __ACTIVATION_FUNCTION__
#define __ACTIVATION_FUNCTION__

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

/**************************** Layer activation ********************************/
/* Softmax activation */
class SoftmaxActivation {
public:
    virtual double computeActivation(const std::size_t neuronIndex, std::vector<double>& all_a);
};

/**************************** Neuron activation *******************************/
/* Parent Class */
class ActivationFunction {
public:
    virtual double computeActivation(const double a) = 0;
    virtual double computeActivationPrime(const double a) = 0;
};

/* Linear activation */
class LinearActivation : public ActivationFunction {
    virtual double computeActivation(const double a);
    virtual double computeActivationPrime(const double a);
};

/* Sigmoid activation */
class SigmoidActivation : public ActivationFunction {
    virtual double computeActivation(const double a);
    virtual double computeActivationPrime(const double a);
};

/* ReLU activation */
class ReLUActivation : public ActivationFunction {
    virtual double computeActivation(const double a);
    virtual double computeActivationPrime(const double a);
};

/******************************** FACTORY *************************************/
class ActivationFunctionFactory {
public:
    static ActivationFunction* Build(const std::string type) {
        if(type == "sigmoid") return new SigmoidActivation();
        if(type == "relu") return new ReLUActivation();
        if(type == "softmax") return new LinearActivation;
        return NULL;
    }
};

#endif

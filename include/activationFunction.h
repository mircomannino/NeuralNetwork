#ifndef __ACTIVATION_FUNCTION__
#define __ACTIVATION_FUNCTION__

#include <iostream>
#include <math.h>

/* Parent Class */
class ActivationFunction {
public:
    virtual double computeActivation(const double a) = 0;
    virtual double computeActivationPrime(const double a) = 0;
};

/* Sigmoid activation */
class SigmoidActivation : public ActivationFunction {
    virtual double computeActivation(const double a);
    virtual double computeActivationPrime(const double a);
};

/******************************** FACTORY *************************************/
class ActivationFunctionFactory {
public:
    static ActivationFunction* Build(const std::string type) {
        if(type == "sigmoid") return new SigmoidActivation();
        return NULL;
    }
};

#endif

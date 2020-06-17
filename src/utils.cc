#include "../include/utils.h"
using namespace std;

/************************** Vector operations *********************************/
double dotProduct(std::vector<double> x1, std::vector<double> x2) {
    // Check if the two vectors has same dimension
    if(x1.size() != x2.size()) return -1;
    // Compute the dot product
    double result = 0.0;
    for(int i = 0; i < x1.size(); i++) {
        result += x1[i] * x2[i];
    }
    return result;
}
/******************************************************************************/

/**************************** Activation functions ****************************/
// Sigmoid
double sigmoid(const double a) {
    return (double(1.0) / double(1.0 + exp(-a)));
}
double sigmoidPrime(const double a) {
    return (sigmoid(a) * (1 - sigmoid(a)));
}
// ReLU
double ReLU(const double a) {
    if(a > 0) return a;
    return 0;
}
double ReLUPrime(const double a) {
    if(a > 0) return 1;
    return 0;
}
/******************************************************************************/
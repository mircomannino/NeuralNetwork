#include "../include/neuron.h"
using namespace std;

/******************************** Class Neuron ********************************/
// Constructor
Neuron::Neuron(const int weightDim, const std::string activationType) {
    // Initialization of the weight in random range [-1, 1]
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    uniform_real_distribution<double> distribution (-1.0, 1.0);
    for(int i = 0; i < weightDim; i++) {
        double random_value = distribution(generator);
        this->weight.push_back(random_value);
        this->gradient.push_back(0.0);
    }
    this->bias = distribution(generator);
    this->gradient.push_back(0.0);  // gradient for the bias term
    this->deltaError = 0.0;
    // Selection of the activation function
    activationFunction = ActivationFunctionFactory::Build(activationType);
}

// Method to compute the output of the neuron
double Neuron::computeOutput(const std::vector<double> x) {
    if(x.size() != this->weight.size()) return false;
    this->activationValue = dotProduct(x, this->weight) + bias;
    return activationFunction->computeActivation(this->activationValue);
}

// Show all tha info of the neuron
void Neuron::printInfo() const {
    cout << "############# Neuron info #################\n";
    cout << "Bias: " << this->bias << endl;
    cout << "Weight[" << this->weight.size() << " x 1]: ";
    for(auto el : this->weight) {
        cout << el << " ";
    }
    cout << endl;
    cout << "Gradient[" << this->gradient.size() << " x 1]: ";
    for(auto el : this->gradient) {
        cout << el << " ";
    }
    cout << endl;
    cout << "Delta error: " << this->deltaError << endl;
    cout << "###########################################\n";
}
/******************************************************************************/

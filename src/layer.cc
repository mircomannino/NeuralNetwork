#include "../include/layer.h"
using namespace std;

/*************************** Class Layer **************************************/
// Constructor
Layer::Layer(const int layerDim, const int weightDim, const std::string activationType) {
    softmaxActivation = (activationType == "softmax") ? (new SoftmaxActivation) : NULL;
    for(int i = 0; i < layerDim; i++) {
        this->neurons.push_back(Neuron(weightDim, activationType));
    }
}

// Method to compute the output of the layer
std::vector<double> Layer::computeOutput(const std::vector<double> x) {
    // Check if the dimensione of the input and the neurons are the same
    if(x.size() != this->getLayerNeuronDim()) return vector<double>{};

    vector<double> result;
    if(softmaxActivation != NULL) {
        std::vector<double> allActivations;
        for(int i = 0; i < this->neurons.size(); i++) {
            allActivations.push_back(this->neurons[i].computeOutput(x));
        }
        for(int i = 0; i < this->neurons.size(); i++) {
            result.push_back(softmaxActivation->computeActivation(i, allActivations));
        }
    } else {
        for(int i = 0; i < this->neurons.size(); i++) {
            result.push_back(this->neurons[i].computeOutput(x));
        }
    }

    return result;
}

// Show the indo of the neuron
void Layer::printInfo() const {
    cout << "###################### Layer info #########################\n";
    cout << "N. of neurons: " << this->neurons.size() << endl;
    cout << "Dimension of each neuron: [" << this->getLayerNeuronDim() << " x 1]\n";
    for(auto neuron: this->neurons) {
        neuron.printInfo();
    }
    cout << "###########################################################\n";
}
/******************************************************************************/

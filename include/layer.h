#ifndef __LAYER__
#define __LAYER__

#include <vector>
#include "neuron.h"

class Layer {
private:
    std::vector<Neuron> neurons;
public:
    Layer() {}
    Layer(const int layerDim, const int weightDim, const std::string activationType);
    inline int getLayerDim() const { return this->neurons.size(); }
    inline int getLayerNeuronDim() const { return this->neurons[0].getWeightDim(); }
    inline std::vector<Neuron> getLayer() const { return this->neurons; }
    inline Neuron& getNeuron(const int index) { return neurons[index]; }
    std::vector<double> computeOutput(std::vector<double> x);
    void printInfo() const;
};
#endif

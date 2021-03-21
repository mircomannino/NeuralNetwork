#include "../include/neuralNetwork.h"
using namespace std;

/*************************** Class NeuralNetwork ******************************/
// Constructor
NeuralNetwork::NeuralNetwork(const std::vector<int> architecture, const std::vector<std::string>& activations) {
    // architecture = [3, 10, 4] --> dimInput:3, dimHidden:10, dimOutput:4
    this->architecture = architecture;
    // Creation of all the layers
    for(int i = 1; i < architecture.size(); i++) {
        Layer newLayer(architecture[i], architecture[i-1], activations[i-1]);
        this->layers.push_back(newLayer);
    }
}

// Method to compute the output of the network
std::vector<double> NeuralNetwork::predict(const vector<double> x) {
    // Check if the input size is equal to the first layer size
    if(x.size() != this->layers[0].getLayerNeuronDim()) return vector<double>{};
    // Compute the result of the network
    vector<double> currentOutput;
    vector<double> currentInput = x;
    for(int i = 0; i < layers.size(); i++) {
        currentOutput = layers[i].computeOutput(currentInput);
        currentInput = currentOutput;
    }
    return currentOutput;
}

// Method to compute the output of the network and save the intermediate output
std::vector<double> NeuralNetwork::predictAndSave(const vector<double> x, vector<vector<double>>& intermediateOutput) {
    // Check if the input size is equal to the first layer size
    if(x.size() != this->layers[0].getLayerNeuronDim()) return vector<double>{};
    // Compute the result of the network
    vector<double> currentOutput;
    vector<double> currentInput = x;
    for(int i = 0; i < layers.size(); i++) {
        currentOutput = layers[i].computeOutput(currentInput);
        intermediateOutput.push_back(currentOutput);
        currentInput = currentOutput;
    }
    return currentOutput;
}

// Method to load weigths from a file
void NeuralNetwork::loadWeigths(const std::string& filePath) {
    std::ifstream is(filePath);
    if(!is.good()) {
        std::cerr << "Error in opening file: " << filePath << std::endl;
        return;
    }

    for(auto& layer : layers) {
        // Get weights of Neurons
        std::string weightsLine_str = "";
        std::getline(is, weightsLine_str);
        std::vector<double> weightsLine = splitString(weightsLine_str);

        // Get bias of neurons
        std::string biasLine_str = "";
        std::getline(is, biasLine_str);
        std::vector<double> biasLine = splitString(biasLine_str);

        // Assign weights and bias to netowrk neurons
        for(size_t i = 0; i < layer.getLayerDim(); i++) {
            for(size_t j = 0; j < layer.getLayerNeuronDim(); j++) {
                layer.getNeuron(i).setWeight(j, weightsLine[i*layer.getLayerNeuronDim() + j]);
            }
            layer.getNeuron(i).setBias(biasLine[i]);
        }
    }


}

// TODO:    Aggiungere il salvataggio di tutte le attivazioni ai, per cambiare il tipo di attivazione in backprop
void NeuralNetwork::train(const std::string& filePath, double learingRate, int nEpochs, int miniBatchSize) {
    // Read the trainingSet from file
    vector<vector<double>> trainingSet = this->loadFromFile(filePath);
    if(trainingSet.size() == 0) {
        cerr << "File not found!\n";
        return;
    }
    // Divide the training set in pattern and target
    vector<vector<double>> trainingTargets;
    vector<vector<double>> trainingPatterns;
    this->dividePatternsAnsTargets(trainingSet, trainingTargets, trainingPatterns);

    //Start the training
    for(int epoch = 0; epoch < nEpochs; epoch++) {
        double error = 0.0;
        for(int kTot = 0; kTot < miniBatchSize; kTot++) {
            int k = ((epoch * miniBatchSize) + kTot) % trainingSet.size();
            vector<vector<double>> X;   // Vector that contains the output of each layer
            // FORWARD STEP
            X.push_back(trainingPatterns[k]);
            vector<double> prediction = this->predictAndSave(trainingPatterns[k], X);

            // ERROR
            for(int i = 0; i < prediction.size(); i++) {
                error += pow((prediction[i] - trainingTargets[k][i]), 2);
                // cout << trainingTargets[k][i] << "-> " << trainingPatterns[k][0] << trainingPatterns[k][1] << ", " << prediction[i] << endl;
            }

            // X:
            // 0 -> input
            // 1 -> hidden
            // 2 -> result

            // h:
            // 0 -> hidden
            // 1 -> output

            // BACKWARD STEP - PER ORA FUNZIONA SOLO CON SIGMOIDE
            // Last layer
            int lastindex = layers.size() - 1;
            for(int i = 0; i < layers[lastindex].getLayerDim(); i++) {
                double activationValue = layers[lastindex].getNeuron(i).getActivationValue();
                double delta_i = 0.0;
                delta_i = prediction[i] - trainingTargets[k][i];
                delta_i *= layers[lastindex].getNeuron(i).getActivationFunction()->computeActivationPrime(activationValue);
                // delta_i *= sigmoidPrime(activationValue);
                layers[lastindex].getNeuron(i).setDeltaError(delta_i);
                for(int j = 0; j < layers[lastindex].getLayerNeuronDim(); j++) {
                    double x_kj = X[lastindex][j];
                    double varGradient_ij = x_kj * delta_i;
                    double oldGradient_ij = layers[lastindex].getNeuron(i).getGradient(j);
                    double newGradient_ij = oldGradient_ij + varGradient_ij;
                    layers[lastindex].getNeuron(i).setGradient(j, newGradient_ij);
                }
                int bias_index = layers[lastindex].getNeuron(i).getWeightDim(); // The last index is tha bias
                double oldGradient_bias = layers[lastindex].getNeuron(i).getGradient(bias_index);
                double newGradient_bias = oldGradient_bias + (1 * delta_i);
                layers[lastindex].getNeuron(i).setGradient(bias_index, newGradient_bias);
            }

            // Hidden layer
            for(int h = layers.size()-2; h >= 0; h--) {
                for(int i = 0; i < layers[h].getLayerDim(); i++) {
                    double sum = 0.0;
                    for(int c = 0; c < layers[h+1].getLayerDim(); c++) {
                        double delta_c = layers[h+1].getNeuron(c).getDeltaError();
                        double weight_ci = layers[h+1].getNeuron(c).getWeight(i);
                        sum += delta_c * weight_ci;
                    }
                    double sigmoid_ki = X[h+1][i];
                    double activationValue = layers[h].getNeuron(i).getActivationValue();
                    double delta_i = layers[h].getNeuron(i).getActivationFunction()->computeActivationPrime(activationValue) * sum;
                    // double delta_i = sigmoidPrime(activationValue) * sum;
                    layers[h].getNeuron(i).setDeltaError(delta_i);
                    for(int j = 0; j < layers[h].getLayerNeuronDim(); j++) {
                        double x_kj = X[h][j];
                        double varGradient_ij = x_kj * delta_i;
                        double oldGradient_ij = layers[h].getNeuron(i).getGradient(j);
                        double newGradient_ij = oldGradient_ij + varGradient_ij;
                        layers[h].getNeuron(i).setGradient(j, newGradient_ij);
                    }
                    int bias_index = layers[h].getNeuron(i).getWeightDim(); // The last index
                    double oldGradient_bias = layers[h].getNeuron(i).getGradient(bias_index);
                    double newGradient_bias = oldGradient_bias + (1 * delta_i);
                    layers[h].getNeuron(i).setGradient(bias_index, newGradient_bias);
                }
            }

        }

        // Update of the weight
        for(int h = 0; h < layers.size(); h++) {
            for(int i = 0; i < layers[h].getLayerDim(); i++) {
                for(int j = 0; j < layers[h].getLayerNeuronDim(); j++) {
                    double oldWeight_ij = layers[h].getNeuron(i).getWeight(j);
                    double gradient_ij = layers[h].getNeuron(i).getGradient(j);
                    double newWeight_ij = oldWeight_ij - (learingRate * gradient_ij);
                    layers[h].getNeuron(i).setWeight(j, newWeight_ij);
                    // Reset the gradient
                    layers[h].getNeuron(i).setGradient(j, 0.0);
                }
                // Update the bias
                int bias_index = layers[h].getNeuron(i).getWeightDim();
                double oldBias_i = layers[h].getNeuron(i).getBias();
                double gradientBias_i = layers[h].getNeuron(i).getGradient(bias_index);
                double newBias_i = oldBias_i - (learingRate * gradientBias_i);
                layers[h].getNeuron(i).setBias(newBias_i);
                // Reset the gradient
                layers[h].getNeuron(i).setGradient(bias_index, 0.0);
            }
        }

        // Print the error
        cout << "Epoch: " << epoch << " ";
        cout << "Error: " << sqrt(error)/miniBatchSize << endl;
    }
}

// Method to test the network
double NeuralNetwork::test(const std::string& filePath) {
    // Read the dataset from file
    vector<vector<double>> testSet = this->loadFromFile(filePath);
    if(testSet.size() == 0) {
        cerr << "File not found\n";
        return -1;
    }

    // Divide the target and the pattern
    vector<vector<double>> testTargets;
    vector<vector<double>> testPatterns;
    this->dividePatternsAnsTargets(testSet, testTargets, testPatterns);

    double correctPercentage = 0.0;

    // Make the test
    for(int k = 0; k < testPatterns.size(); k++) {
        vector<double> prediction = this->predict(testPatterns[k]);
        cout << "Real: ";
        for(auto el : testTargets[k]) cout << std::setprecision(2) << el << " ";
        cout << " - Predicted: ";
        for(auto el : prediction) cout << std::setprecision(2) << el << " ";
        cout << endl;
    }

    return correctPercentage;
}

// Method to load training set or test set from file
// line file format:    target0, target1, ..., pattern0, pattern1, ...
std::vector<std::vector<double>> NeuralNetwork::loadFromFile(const string& filePath) {
    ifstream fileStream(filePath);
    if(!fileStream.good()) return vector<vector<double>>{};

    vector<vector<double>> result;
    int dimInput = architecture[0];
    int dimOutput = architecture[architecture.size()-1];

    string line = "";
    while(getline(fileStream, line)) {
        vector<double> lineResult;
        istringstream lineStream(line);
        // Get the target
        for(int i = 0; i < dimOutput; i++) {
            double valTarget = 0.0;
            lineStream >> valTarget;
            lineResult.push_back(valTarget);
        }
        // Get the pattern
        for(int i = 0; i < dimInput; i++) {
            double valPattern = 0.0;
            lineStream >> valPattern;
            lineResult.push_back(valPattern);
        }
        result.push_back(lineResult);
    }
    return result;
}

// Method to divide targets and patterns of a given training/test set
// input line data:     target0, target1, ..., pattern0, pattern1, ...
// output line data:    [target0, target1, ...], [pattern0, pattern1, ...]
void NeuralNetwork::dividePatternsAnsTargets(
    std::vector<std::vector<double>> data,
    std::vector<std::vector<double>>& targets,
    std::vector<std::vector<double>>& patterns) {

    targets = vector<vector<double>> {};
    patterns = vector<vector<double>>{};

    for(int i = 0; i < data.size(); i++) {
        int dimOutput = architecture[architecture.size()-1];

        vector<double> singleTarget;
        for(int j = 0; j < dimOutput; j++) {
            singleTarget.push_back(data[i][j]);
        }
        targets.push_back(singleTarget);

        vector<double> singlePattern;
        for(int j = dimOutput; j < data[i].size(); j++) {
            singlePattern.push_back(data[i][j]);
        }
        patterns.push_back(singlePattern);
    }
}



// Show the information of the NeuralNetwork
void NeuralNetwork::printInfo() const {
    cout << "########################### Neural netork ######################\n";
    cout << "N. of Layer: " << this->layers.size() << endl;
    for(auto layer : layers) {
        layer.printInfo();
    }
}
/******************************************************************************/

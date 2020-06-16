# Define parameters
CC=g++

all: testXOR testIRIS

# Test for XOR dataset
testXOR: utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testXOR.o
	$(CC) -o testXOR utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testXOR.o

# Test for IRIS dataset
testIRIS: utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testIRIS.o
	$(CC) -o testIRIS utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testIRIS.o

utils.o: utils.cc
	$(CC) -c utils.cc

activationFunction.o: activationFunction.cc
	${CC} -c activationFunction.cc

neuron.o: neuron.cc
	$(CC) -c neuron.cc

layer.o: layer.cc
	$(CC) -c layer.cc

neuralNetwork.o: neuralNetwork.cc
	$(CC) -c neuralNetwork.cc

testXOR.o: testXOR.cc
	$(CC) -c testXOR.cc

clean:
	rm -rf *.o testXOR testIRIS

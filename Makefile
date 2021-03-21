# Define parameters
CC=g++
SRC=./src

all: testXOR testIRIS testAUTOENCODER testTime

# Test for XOR dataset
testXOR: utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testXOR.o
	$(CC) -o testXOR utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testXOR.o 

# Test for IRIS dataset
testIRIS: utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testIRIS.o
	$(CC) -o testIRIS utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testIRIS.o

# Test for Autoencoder
testAUTOENCODER: utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testAUTOENCODER.o
	$(CC) -o testAUTOENCODER utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testAUTOENCODER.o

testTime: utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testTime.o
	$(CC) -o testTime utils.o activationFunction.o neuron.o layer.o neuralNetwork.o testTime.o

utils.o: $(SRC)/utils.cc
	$(CC) -c $(SRC)/utils.cc

activationFunction.o: $(SRC)/activationFunction.cc
	${CC} -c $(SRC)/activationFunction.cc

neuron.o: $(SRC)/neuron.cc
	$(CC) -c $(SRC)/neuron.cc

layer.o: $(SRC)/layer.cc
	$(CC) -c $(SRC)/layer.cc

neuralNetwork.o: $(SRC)/neuralNetwork.cc
	$(CC) -c $(SRC)/neuralNetwork.cc

testXOR.o: $(SRC)/testXOR.cc
	$(CC) -c $(SRC)/testXOR.cc

testIRIS.o: $(SRC)/testIRIS.cc
	$(CC) -c $(SRC)/testIRIS.cc

testAUTOENCODER.o: $(SRC)/testAUTOENCODER.cc
	$(CC) -c $(SRC)/testAUTOENCODER.cc

testTime.o: ${SRC}/testTime.cc
	${CC} -c ${SRC}/testTime.cc

clean:
	rm -rf *.o testXOR testIRIS testAUTOENCODER testTime

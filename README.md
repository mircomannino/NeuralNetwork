# Neural Network: C++ implementation
In this repository you will find a implementation of a feedforward fully-connected
network.

## Compile
To compile the program run the following command:
```
make
```
It will create the executable file:
* testXOR
* testIRIS  

To remove all the files created by _make_ command run the following command:
```
make clean
```

## Test XOR
To execute the test based on the XOR dataset run the following command:
```
./testXOR
```
The output should be something like:
```
Real: 0  - Predicted: 0.044
Real: 1  - Predicted: 0.93
Real: 1  - Predicted: 0.94
Real: 0  - Predicted: 0.082
```

## Test IRIS
To execute the test based in the IRIS dataset run the following command:
```
./testIRIS
```

## Activation functions
For the activation functions was used the __sigmoid function__. It can be found in _utils.cc_.

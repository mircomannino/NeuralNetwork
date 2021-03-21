#ifndef __UTILS__
#define __UTILS__

#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

double dotProduct(std::vector<double> x1, std::vector<double> x2);
int argMax(const std::vector<double>& v);
void printVector(const std::vector<double>& v);
std::vector<double> splitString(const std::string& inputString);

#endif

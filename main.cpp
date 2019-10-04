#include <iostream>
#include "MatrixOpApi.hpp"
#include "MatrixTest.hpp"
#include "ThreeLayerNNApi.hpp"
#include "ThreeLayerNNTest.hpp"
using namespace std;
int main() {
	test_matrix_compare_and_cal_error_rate();
	return 0;
}


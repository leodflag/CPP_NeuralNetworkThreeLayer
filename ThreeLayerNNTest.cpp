#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h> 
#include <math.h>
#include "ThreeLayerNNApi.hpp"
#include "MatrixOpApi.hpp"
void test_matrix_sigmoid_der(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=matrix_sigmoid(A);
	printData(B);	
	Matrix C=matrix_sigmoid_der(B);
	printData(C);	
}
void test_matrix_loss_function_der(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=create_one_matrix(r,c);
	printData(B);
	Matrix C=matrix_loss_function(A,B);
	printData(C);	
	Matrix D=matrix_loss_function_der(A,B);
	printData(D);	
}
void test_create_net_layer(){
	Net_layer hidden_layer=create_net_layer(2,3,2);
}
void test_one_hot_encoding(){
	Matrix data=create_rand_matrix(4,3);
	printData(data);
	Matrix lable=matrix_get_col_lable_data(data,2);
	Matrix A=one_hot_encoding(data,lable);
	printData(A);
	Matrix D=create_new_matrix(4,3);
	read_matrix_data(D);
	printData(D);
	Matrix L=matrix_get_col_lable_data(D,2);
	Matrix AA=one_hot_encoding(D,L);
	printData(AA);
}
void test_nn_api_all(){
	test_create_net_layer();
	test_one_hot_encoding();
}

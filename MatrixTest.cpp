#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h> 
#include <math.h>
#include "MatrixOpApi.hpp"
#include "MatrixTest.hpp"
using namespace std;
void test_create_rand_matrix(){ // OK
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
}
void test_create_one_matrix(){ //OK
	int r=2,c=3;
	Matrix A=create_one_matrix(r,c);
	printData(A);
}
void test_create_zero_matrix(){ //OK
	int r=4,c=3;
	Matrix A=create_zero_matrix(r,c);
	printData(A);
}
void test_matrix_tran_last_col_negative(){
	int r=2,c=3;
	Matrix A=create_one_matrix(r,c);
	Matrix B=matrix_tran_last_col_negative(A);
	printData(B);
}
void test_matrix_transpose(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	Matrix B=matrix_transpose(A);
	printData(B);
}
void test_matrix_plus(){
	int r=2,c=2;
	Matrix A=create_one_matrix(r,c);
	Matrix B=create_one_matrix(r,c);
	Matrix C=matrix_plus(A,B);
	printData(C);
}
void test_matrix_sub(){
	int r=2,c=2;
	Matrix A=create_one_matrix(r,c);
	Matrix B=create_one_matrix(r,c);
	Matrix C=matrix_sub(A,B);
	printData(C);
}
void test_matrix_mult(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=create_one_matrix(r,c);
	printData(B);
	Matrix C=matrix_mult(A,B);
	printData(C);
}
void test_matrix_hadamard(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=create_one_matrix(r,c);
	printData(B);
	Matrix C=matrix_hadamard(A,B);
	printData(C);
}
void test_matrix_sigmoid(){
	int r=2,c=2;
	Matrix A=create_rand_matrix(r,c);
	Matrix B=matrix_sigmoid(A);
	printData(B);
}
void test_matrix_sigmoid_der(){
	int r=2,c=2;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=matrix_sigmoid_der(A);
	printData(B);
}
void test_matrix_total_num(){
	int r=2,c=2;
	Matrix A=create_one_matrix(r,c);
	printData(A);
	double a=matrix_total_num(A);
	printf("All=%f",a);
}
void test_matrix_loss_function(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=create_one_matrix(r,c);
	printData(B);
	Matrix C=matrix_loss_function(A,B);
	printData(C);	
}
void test_matrix_loss_function_der(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=create_one_matrix(r,c);
	printData(B);
	Matrix C=matrix_loss_function_der(A,B);
	printData(C);	
}


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
void test_read_matrix_data(){ 
	Matrix A=create_new_matrix(4,3);
	read_matrix_data(A);
	printData(A);
}
void test_create_rand_matrix(){ 
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix A1=create_rand_matrix(r,c);
	printData(A1);
}
void test_create_one_matrix(){ 
	int r=3,c=1;
	Matrix A=create_one_matrix(r,c);
	printData(A);
}
void test_create_zero_matrix(){
	int r=4,c=3;
	Matrix A=create_zero_matrix(r,c);
	printData(A);
}
void test_matrix_equal(){
	int r=2,c=3;
	Matrix A=create_one_matrix(r,c);
	Matrix B=matrix_equal(A);
	printData(B);
}
void test_matrix_find_max(){
	Matrix A=create_rand_matrix(4,3);
	printData(A);
	Matrix B=matrix_find_max(A);
	printData(B);
}
void test_matrix_find_max_col(){
	Matrix A=create_rand_matrix(1,3);
	printData(A);
	int B=matrix_find_max_col(A);
	printf("B=%d:",B);
}
void test_matrix_compare_and_cal_error_rate(){
	Matrix A=create_rand_matrix(8,1);
	printData(A);
	Matrix B=matrix_equal(A);
	B.data_matrix[1][0]=0.66;
	printData(B);
	double err_rate=matrix_compare_and_cal_error_rate(A,B);
	printf("error=%f\n",err_rate);
}
void test_matrix_tran_last_col_negative(){
	int r=2,c=3;
	Matrix A=create_one_matrix(r,c);
	Matrix B=matrix_tran_last_col_negative(A);
	printData(B);
}
void test_matrix_add_col_one(){
	int r=2,c=3;
	Matrix A=create_one_matrix(r,c);
	Matrix B=matrix_add_col_one(A);
	printData(B);
}
void test_matrix_delete_last_col_data(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=matrix_delete_last_col_data(A);
	printData(B);
}
void test_matrix_get_multi_row_data(){
	int r=10,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	printf("-----A------\n\n");
	Matrix B=matrix_get_multi_row_data(A,2,8);
	printData(B);
}
void test_matrix_get_one_row_data(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=matrix_get_one_row_data(A,0);
	printData(B);
}
void test_matrix_get_col_label_data(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=matrix_get_col_label_data(A,2);
	printData(B);
}
void test_matrix_row_sort_small_to_large(){
	int r=2,c=10,n=0;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=matrix_row_sort_small_to_large(A,n);
	printData(B);
}
void test_matrix_transpose(){
	int r=2,c=3;
	Matrix A=create_rand_matrix(r,c);
	Matrix B=matrix_transpose(A);
	printData(B);
}
void test_matrix_random_order(){
	int r=10,c=2;
	Matrix A=create_rand_matrix(r,c);
	Matrix B=matrix_random_order(A);
	printData(A);
	printf("-----A------\n\n");
	printData(B);
}
void test_matrix_plus(){
	int r=2,c=3;
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
void test_matrix_mult_num(){
	int r=1,c=4;
	double x=2.0;
	Matrix A=create_one_matrix(r,c);
	A=matrix_mult_num(A,x);
	printData(A);
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
void test_matrix_total(){
	int r=2,c=2;
	Matrix A=create_one_matrix(r,c);
	printData(A);
	double a=matrix_total(A);
	printf("All=%f",a);
}
void test_matrix_all(){
	test_read_matrix_data();
	test_create_rand_matrix();
	test_create_one_matrix();
	test_create_zero_matrix();
	test_matrix_tran_last_col_negative();
	test_matrix_add_col_one();
	test_matrix_delete_last_col_data();
	test_matrix_get_one_row_data();
	test_matrix_row_sort_small_to_large();
	test_matrix_get_col_label_data();
	test_matrix_transpose();
	test_matrix_plus();
	test_matrix_sub();
	test_matrix_mult();
	test_matrix_mult_num();
	test_matrix_hadamard();
	test_matrix_total();
}     

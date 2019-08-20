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
void test_label_processing(){
	Matrix Data=create_new_matrix(4,3);
	read_matrix_data(Data);
	Matrix label=label_processing(Data);
	printData(label);
}
void test_data_processing(){
	Matrix Data=create_new_matrix(4,3);
	read_matrix_data(Data);
	Data=data_processing(Data);
	printData(Data);
}
void test_matrix_hidden_layer_error(){
	int r=2,c=3,rb=1,cb=2;
	Matrix A=create_rand_matrix(r,c);
	printData(A);
	Matrix B=create_rand_matrix(rb,cb);
	printData(B);
	Matrix C= matrix_hidden_layer_error(A,B);
	printData(C);
}
void test_create_net_layer(){
	Net_layer hidden_layer=create_net_layer(2,3,2);
}
void test_one_hot_encoding(){
	Matrix data=create_rand_matrix(4,3);
	printData(data);
	Matrix label=matrix_get_col_label_data(data,2);
	Matrix A=one_hot_encoding(data,label);
	printData(A);
	Matrix D=create_new_matrix(4,3);
	read_matrix_data(D);
	printData(D);
	Matrix L=matrix_get_col_label_data(D,2);
	Matrix AA=one_hot_encoding(D,L);
	printData(AA);
}
void test_net_update_weight(){
	double learning_rate=2.0;
	int r=2,c=3,data_order=0;
	NeuralNetwork NN;
	NN.H_layer=create_net_layer(1,3,2);
	NN.O_layer=create_net_layer(1,3,2);
	NN.H_layer.w=create_new_matrix(r,c);
	printData(NN.H_layer.w);
	NN.H_layer.error=create_rand_matrix(r,c);
	NN.O_layer.error=create_rand_matrix(r,c);
	NN.H_layer.net_sigmoid=create_rand_matrix(1,2);
	Matrix Data=create_new_matrix(4,3);
	read_matrix_data(Data);
	NN=net_update_weight(NN,learning_rate,Data,data_order);
//	int R=2,C=3;
//	Matrix A=create_new_matrix(R,C);
//	printData(A);
//	Matrix B=create_rand_matrix(R,C);
//	printData(B);
//	Matrix CC=create_rand_matrix(R,C);
//	printData(CC);
//	for(int r=0;r<R;r++){
//		for(int c=0;c<C-1;c++){
//			A.data_matrix[r][c]=-learning_rate*B.data_matrix[0][r]*CC.data_matrix[0][c];
//		}
//	}
//	printData(A);
//	NN.O_layer.delta_w.data_matrix[r][c]=-learning_rate*NN.O_layer.error.data_matrix[0][r]*NN.H_layer.net_sigmoid.data_matrix[0][c];
	
}
void test(){
	int data_order=0,n=1;
	double Learning_r=2.0;
	NeuralNetwork NN;
	NN.H_layer=create_net_layer(1,3,2);
	NN.O_layer=create_net_layer(1,3,2);
	Matrix Data=create_new_matrix(4,3);
	read_matrix_data(Data);
	Matrix Label=label_processing(Data);
	Data=data_processing(Data);
//	printData(Label);
	while(n>0){
//		while(data_order<4){
		NN=net_forward(NN,Data,data_order);
		printALLData(NN);
		Matrix Label_1=matrix_get_one_row_data(Label,data_order);
//		printData(NN.O_layer.net_sigmoid);
//		printData(Label_1);
//		printf("\n\n\n");
		Matrix ERROR=matrix_loss_function_der(Label_1,NN.O_layer.net_sigmoid);
//		printData(ERROR);
//		printf("LABET\n\n\n");11
		NN=net_back(NN,Data,Label_1,data_order);
//		printALLData(NN);
//			printf("\n back\n");
		
		NN=net_update_weight(NN,Learning_r,Data,data_order);
//		printALLData(NN);
		NN=net_update_bais(NN,Learning_r);	
//		printALLData(NN);
	//		printData(NN.O_layer.net_sigmoid);
//			printf("\n end------%d \n\n",data_order);
	//		printData(NN.O_layer.error);
	//		printf("\n\n");
//			data_order++;		
//		}
//		data_order=0;
		n--;
	}
}

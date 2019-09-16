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
	Matrix Data=create_new_matrix(12,3);
	read_matrix_data(Data);
	printData(Data);
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
	int r=2,c=3,data_order=3;
	NeuralNetwork NN;
	NN.H_layer=create_net_layer(1,3,2);
	NN.O_layer=create_net_layer(1,3,2);
	NN.H_layer.w=create_new_matrix(r,c);
	NN.H_layer.error=create_rand_matrix(r,c);
	NN.O_layer.error=create_rand_matrix(r,c);
	NN.H_layer.net_sigmoid=create_rand_matrix(1,2);
	Matrix Data=create_new_matrix(4,3);
	read_matrix_data(Data);
	NN=net_update_weight(NN,learning_rate,Data);
	printALLData(NN);
}
void test_SGD(){
//	double learningRate=2.0;
	// data_four.csv �w�����T�ƭ� 
//	Matrix Data=create_new_matrix(12,3);
//	read_matrix_data(Data);
//	SGD(Data,3,4,3,2.0,1000);
	// data.csv
//	Matrix Data=create_new_matrix(4,3);
//	read_matrix_data(Data);
//	SGD(Data,2,2,3,2.0,1000);
	// data_iris.csv
	Matrix Data=create_new_matrix(150,5);
	read_matrix_data(Data);
	SGD(Data,2,3,5,2.0,100);
}
void test(){
	Matrix Data=create_new_matrix(150,5);
	read_matrix_data(Data);
	int hidden_net_num=2;
	int output_net_num=3;
	int data_col=5;
	double learning_rate=1.3;
	int iteration=1000;
	int data_order=0;
	NeuralNetwork NN;
	NN.H_layer=create_net_layer(1,data_col,hidden_net_num); // �إ����üh 
	NN.O_layer=create_net_layer(1,hidden_net_num+1,output_net_num); // ��X�h�M���üh�x�}�B��ɡAcol�n�[�Wbais 
	Matrix Label=label_processing(Data); // label �B�z 
	Data=data_processing(Data); // data �B�z 
	while(iteration>0){ // �`������ 
		while(data_order<Data.data_row){ // �`���@������� 
			Matrix DATA=matrix_get_one_row_data(Data,data_order); // ���o�@����� 
			NN=net_forward(NN,DATA); // �e�V�Ǽ� 
//			printData(NN.O_layer.net_sigmoid)
			Matrix Label_1=matrix_get_one_row_data(Label,data_order); // ���o�P�C��label 
//			if(iteration==1) // �u�L�X�̫�@�����w�����G 
//				printData(NN.O_layer.net_sigmoid);
			Matrix ERROR=matrix_loss_function(Label_1,NN.O_layer.net_sigmoid); // �p��error
			if(iteration==1)
				printData(ERROR);
			NN=net_back(NN,Label_1); // �˶ǻ� 
			NN=BGD_calculate_delta_weight(NN,learning_rate,DATA); //�p��C���ƭȪ��v���Pbais�å[�_�� 
//			printALLData(NN);
//			NN=net_update_weight(NN,learning_rate,DATA); // �p��ç�s�v�� 
//			NN=net_update_bais(NN,learning_rate);  // �p��ç�sbais 
			data_order++;		
		}
		NN=BGD_update_weight_and_bais(NN,4);
//		printf("------iteration=%d------\n",iteration);
		data_order=0;
		iteration--;
	}
}

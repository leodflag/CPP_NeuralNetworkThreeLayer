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
Matrix matrix_sigmoid(Matrix Data){ // �Ҧ��x�}�ƭȬҹLsigmoid���
	Matrix Data_sig=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_sig.data_matrix[r][c]=1.0/(1.0+exp(-Data.data_matrix[r][c]));
		}
	}
	return Data_sig;
}
Matrix matrix_sigmoid_der(Matrix Data){ // �Ҧ��x�}�ƭȬҹLsigmoid�ɨ��
	Matrix Data_sig_der=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_sig_der.data_matrix[r][c]=Data.data_matrix[r][c]*(1.0-Data.data_matrix[r][c]);
		}
	}
	return Data_sig_der;
}
Matrix matrix_loss_function(Matrix Matrix_tag,Matrix Matrix_out){ // �l�����( �ؼЯx�}�A��X�x�} )
	Matrix OUT=create_new_matrix(Matrix_out.data_row,Matrix_out.data_col);
	double ans;
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			ans=0.0;
			ans=(Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c]);
			OUT.data_matrix[r][c]=(1.0/2.0)*pow(ans,2.0);
		}
	}
	return OUT;
}
Matrix matrix_loss_function_der(Matrix Matrix_tag,Matrix Matrix_out){ // �l���ɨ��( �ؼЯx�}�A��X�x�} )
	Matrix OUT=create_new_matrix(Matrix_out.data_row,Matrix_out.data_col);
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			OUT.data_matrix[r][c]=Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c];
		}
	}
	return OUT;
}
Matrix label_processing(Matrix Data){
	Matrix Label=matrix_get_col_label_data(Data,Data.data_col-1);
//	printData(Label);
	Label=one_hot_encoding(Data,Label);
	return Label;
}
Matrix data_processing(Matrix Data){
	Data=matrix_delete_last_col_data(Data);
	Data=matrix_add_col_one(Data); 
	return Data;
}
Net_layer create_net_layer(int data_num,int col,int net_num){ // �سy���g�h�A��J�`��ƾ��ӼơA����ӼơA���g���Ӽ�  
	Net_layer NetLayer;
	NetLayer.w=create_rand_matrix(net_num,col);  // �إ����üh����l�v���x�}  
	NetLayer.delta_w=create_zero_matrix(net_num,col);  // �إ߭ץ��v���x�} 
	NetLayer.net=create_zero_matrix(data_num,net_num);
	NetLayer.net_sigmoid=create_zero_matrix(data_num,net_num);
	NetLayer.error=create_zero_matrix(data_num,net_num);
	return NetLayer;
}
Matrix matrix_hidden_layer_error(Matrix weight,Matrix error){
	Matrix Matrix_A=create_new_matrix(error.data_row,error.data_col);
	for(int c=0;c<Matrix_A.data_col;c++){ //2
		for(int k=0;k<weight.data_col-1;k++){ //2
			Matrix_A.data_matrix[0][c]+=error.data_matrix[0][k]*weight.data_matrix[k][c];
		}
	}
	return Matrix_A;
}
Matrix one_hot_encoding(Matrix data,Matrix label){  // ���쥻��data,��X��label 
	Matrix label_1=matrix_row_sort_small_to_large(label,0);// �N����{1,0}���ǧ令{0,1) 	
//	printData(label_1);
	Matrix goal_matrix=create_new_matrix(data.data_row,label.data_col);
	for(int i=0;i<data.data_row;i++){
		for(int j=0;j<goal_matrix.data_col;j++){
				if(data.data_matrix[i][data.data_col-1]==label_1.data_matrix[0][j])
					goal_matrix.data_matrix[i][j]=1.0;
				else
					goal_matrix.data_matrix[i][j]=0.0;
		}
	}
	return goal_matrix;
}
NeuralNetwork net_forward(NeuralNetwork NN,Matrix Data){
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w); // �Nbais�ন�t�� 
//	printData(NN.H_layer.w);
//	cout<<"i***********bais�ন�t��***********\n"<<endl;
	NN.H_layer.net=matrix_mult(Data,NN.H_layer.w); // �x�}���k�A���N�v����m 
//	printData(NN.H_layer.net);
//	cout<<"i***********�x�}���k�A���N�v����m***********\n"<<endl;
	NN.H_layer.net_sigmoid=matrix_sigmoid(NN.H_layer.net);
//	printData(NN.H_layer.net_sigmoid);
//	cout<<"i***********hidden_net***********\n"<<endl;
	Matrix D=matrix_add_col_one(NN.H_layer.net_sigmoid);
//	printData(D);	
//	cout<<"i**********-*�[�Fbais***********\n"<<endl;	
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); 
//	printData(NN.O_layer.w);
//	cout<<"i***********bais�ন�t��***********\n"<<endl;
	NN.O_layer.net=matrix_mult(D,NN.O_layer.w); // �x�}���k�A���N�v����m 
//	printData(NN.O_layer.net);
//	cout<<"i***********Output_net***********\n"<<endl;
	NN.O_layer.net_sigmoid=matrix_sigmoid(NN.O_layer.net);
//	printData(NN.O_layer.net_sigmoid);	
//	cout<<"i***********�w����***********\n"<<endl; 
	return NN;	
} 
NeuralNetwork net_back(NeuralNetwork NN,Matrix Label){   //OK
	NN.O_layer.error=matrix_loss_function_der(Label,NN.O_layer.net_sigmoid); // �q��X�h���^�� (T-Y) 
//	printData(NN.O_layer.net_sigmoid);
//	printData(Label);
//	printData(NN.O_layer.error);
//	cout<<"i***********����P��X�����~�t***********\n"<<endl; 
	Matrix Sigmoid_der=matrix_sigmoid_der(NN.O_layer.net_sigmoid);  // Sigmoid �ɨ�� Y(1-Y)
//	printData(Sigmoid_der);
//	cout<<"i***********��X�h�w����sigmoid�ɨ��***********\n"<<endl;
	NN.O_layer.error=matrix_hadamard(NN.O_layer.error,Sigmoid_der); //�ϥΫ��F���n�x�}���k (T-Y)Y(1-Y) 
//	printData(NN.O_layer.error); 
//	printData(NN.O_layer.w); 
//	cout<<"i***********��X�h�~�t***********\n"<<endl;
//	cout<<"i***********��hbais����X�h�v��***********\n"<<endl; 
	Matrix Matrix_H_err=matrix_hidden_layer_error(NN.O_layer.w,NN.O_layer.error);
//	printData(Matrix_H_err);
//	cout<<"i***********�v�������~�ۥ[***********\n"<<endl;/
	NN.H_layer.error=matrix_sigmoid_der(NN.H_layer.net_sigmoid);  
//	printData(Matrix_H_err); 
//	printData(NN.H_layer.error);
//	cout<<"i***********���üh�w����sigmoid�ɨ��***********\n"<<endl;
	NN.H_layer.error=matrix_hadamard(NN.H_layer.error,Matrix_H_err);
//	printData(NN.H_layer.error);
//	cout<<"i***********���üh�~�t***********\n"<<endl;
    return NN;
}
NeuralNetwork net_update_weight(NeuralNetwork NN,double learning_rate,Matrix Data){
	for(int r=0;r<NN.O_layer.delta_w.data_row;r++){
		for(int c=0;c<NN.O_layer.delta_w.data_col-1;c++){
			NN.O_layer.delta_w.data_matrix[r][c]=learning_rate*NN.O_layer.error.data_matrix[0][r]*NN.H_layer.net_sigmoid.data_matrix[0][c];
		}
	}
	for(int r=0;r<NN.H_layer.delta_w.data_row;r++){
		for(int c=0;c<NN.H_layer.delta_w.data_col-1;c++){
			NN.H_layer.delta_w.data_matrix[r][c]=learning_rate*NN.H_layer.error.data_matrix[0][r]*Data.data_matrix[0][c];
		}
	}
	NN.O_layer.w=matrix_plus(NN.O_layer.delta_w,NN.O_layer.w);
	NN.H_layer.w=matrix_plus(NN.H_layer.delta_w,NN.H_layer.w);
	
	return NN;
}
NeuralNetwork net_update_bais(NeuralNetwork NN,double learning_rate){
	NN.H_layer.w=matrix_tran_last_col_negative(NN.H_layer.w);
	NN.O_layer.w=matrix_tran_last_col_negative(NN.O_layer.w); 
	for(int r=0;r<NN.O_layer.delta_w.data_row;r++){
		NN.O_layer.delta_w.data_matrix[r][NN.O_layer.delta_w.data_col-1]=-learning_rate*NN.O_layer.error.data_matrix[0][r];
		
	}
	for(int r=0;r<NN.H_layer.delta_w.data_row;r++){
		NN.H_layer.delta_w.data_matrix[r][NN.H_layer.delta_w.data_col-1]=-learning_rate*NN.H_layer.error.data_matrix[0][r];
	}
	NN.O_layer.w=matrix_plus(NN.O_layer.delta_w,NN.O_layer.w);
	NN.H_layer.w=matrix_plus(NN.H_layer.delta_w,NN.H_layer.w);
	NN.H_layer.delta_w=re_zero(NN.H_layer.delta_w);
	NN.O_layer.delta_w=re_zero(NN.O_layer.delta_w);
	NN.H_layer.net=re_zero(NN.H_layer.net);
	NN.O_layer.net=re_zero(NN.O_layer.net);
	NN.H_layer.net_sigmoid=re_zero(NN.H_layer.net_sigmoid);
	NN.O_layer.net_sigmoid=re_zero(NN.O_layer.net_sigmoid);
	NN.H_layer.error=re_zero(NN.H_layer.error);
	NN.O_layer.error=re_zero(NN.O_layer.error);
	return NN;	
}
void printALLData(NeuralNetwork NN){
	printData(NN.H_layer.delta_w);
	printData(NN.H_layer.error);
	printData(NN.H_layer.net);
	printData(NN.H_layer.net_sigmoid);
	printData(NN.H_layer.w);
	printf("------H-------\n");
	printData(NN.O_layer.delta_w);
	printData(NN.O_layer.error);
	printData(NN.O_layer.net);
	printData(NN.O_layer.net_sigmoid);
	printData(NN.O_layer.w);
	printf("------O-------\n");
}

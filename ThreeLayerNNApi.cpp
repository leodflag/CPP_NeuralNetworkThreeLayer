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
Matrix matrix_sigmoid(Matrix Data){ // 所有矩陣數值皆過sigmoid函數
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data.data_matrix[r][c]=1/(1+exp(-Data.data_matrix[r][c]));
		}
	}
	return Data;
}
Matrix matrix_sigmoid_der(Matrix Data){ // 所有矩陣數值皆過sigmoid導函數
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data.data_matrix[r][c]*=(1-Data.data_matrix[r][c]);
		}
	}
	return Data;
}
Matrix matrix_loss_function(Matrix Matrix_tag,Matrix Matrix_out){ // 損失函數( 目標矩陣，輸出矩陣 )
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			Matrix_out.data_matrix[r][c]=(1.0/2.0)*pow(Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c],2.0);
		}
	}
	return Matrix_out;
}
Matrix matrix_loss_function_der(Matrix Matrix_tag,Matrix Matrix_out){ // 損失導函數( 目標矩陣，輸出矩陣 )
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			Matrix_out.data_matrix[r][c]=2*(1.0/2.0)*(Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c]);
		}
	}
	return Matrix_out;
}
Net_layer create_net_layer(int data_num,int col,int net_num){ // 建造神經層，輸入總資料橫行個數，直行個數，神經元個數  
	Net_layer NetLayer;
	NetLayer.w=create_rand_matrix(net_num,col);  // 建立隱藏層的初始權重矩陣  
	NetLayer.delta_w=create_new_matrix(net_num,col);  // 建立修正權重矩陣 
	NetLayer.net=create_new_matrix(net_num,data_num);
	NetLayer.net_sigmoid=create_new_matrix(net_num,data_num);
	NetLayer.error=create_new_matrix(net_num,data_num);
	return NetLayer;
}
Matrix one_hot_encoding(Matrix data,Matrix lable){  // 給原本的data,抓出的lable 
	Matrix lable_1=matrix_row_sort_small_to_large(lable,0);// 將標籤{1,0}順序改成{0,1) 	
	printData(lable_1);
	Matrix goal_matrix=create_new_matrix(data.data_row,lable.data_col);
	for(int i=0;i<goal_matrix.data_row;i++){
		for(int j=0;j<goal_matrix.data_col;j++){
				if(data.data_matrix[i][data.data_col-1]==lable_1.data_matrix[0][j])
					goal_matrix.data_matrix[i][j]=1;
				else
					goal_matrix.data_matrix[i][j]=0;
		}
	}
	return goal_matrix;
}

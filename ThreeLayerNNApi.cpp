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
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data.data_matrix[r][c]=1/(1+exp(-Data.data_matrix[r][c]));
		}
	}
	return Data;
}
Matrix matrix_sigmoid_der(Matrix Data){ // �Ҧ��x�}�ƭȬҹLsigmoid�ɨ��
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data.data_matrix[r][c]*=(1-Data.data_matrix[r][c]);
		}
	}
	return Data;
}
Matrix matrix_loss_function(Matrix Matrix_tag,Matrix Matrix_out){ // �l�����( �ؼЯx�}�A��X�x�} )
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			Matrix_out.data_matrix[r][c]=(1.0/2.0)*pow(Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c],2.0);
		}
	}
	return Matrix_out;
}
Matrix matrix_loss_function_der(Matrix Matrix_tag,Matrix Matrix_out){ // �l���ɨ��( �ؼЯx�}�A��X�x�} )
	for(int r=0;r<Matrix_tag.data_row;r++){
		for(int c=0;c<Matrix_tag.data_col;c++){
			Matrix_out.data_matrix[r][c]=2*(1.0/2.0)*(Matrix_tag.data_matrix[r][c]-Matrix_out.data_matrix[r][c]);
		}
	}
	return Matrix_out;
}
Net_layer create_net_layer(int data_num,int col,int net_num){ // �سy���g�h�A��J�`��ƾ��ӼơA����ӼơA���g���Ӽ�  
	Net_layer NetLayer;
	NetLayer.w=create_rand_matrix(net_num,col);  // �إ����üh����l�v���x�}  
	NetLayer.delta_w=create_new_matrix(net_num,col);  // �إ߭ץ��v���x�} 
	NetLayer.net=create_new_matrix(net_num,data_num);
	NetLayer.net_sigmoid=create_new_matrix(net_num,data_num);
	NetLayer.error=create_new_matrix(net_num,data_num);
	return NetLayer;
}
Matrix one_hot_encoding(Matrix data,Matrix lable){  // ���쥻��data,��X��lable 
	Matrix lable_1=matrix_row_sort_small_to_large(lable,0);// �N����{1,0}���ǧ令{0,1) 	
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

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h> 
#include <math.h>
#include "MatrixOpApi.hpp"
using namespace std;
Matrix create_new_matrix(int r,int c){ // 建立動態指標
	Matrix Data;
	Data.data_row=r;
	Data.data_col=c;
	Data.data_matrix=NULL; //宣告矩陣
	Data.data_matrix=new double *[Data.data_row]; //建立有data_row個string的陣列位址
	for(int i=0;i<Data.data_row;i++)
		Data.data_matrix[i]=new double[Data.data_col]; // 每條陣列位址內再加data_col個string的陣列位址
	return Data;
}
void printData(Matrix Data){
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			cout<<Data.data_matrix[r][c]<<",";
		}
		printf("\n");
	}printf("-----------\n");
}
Matrix creatMatrix(double A[][3],int r,int c){
    Matrix Data=create_new_matrix(r,c);
    for(int i=0;i<Data.data_row;i++){
        for(int j=0;j<Data.data_col;j++)
            Data.data_matrix[i][j]=A[i][j];
    }	
    return Data;
} 
Matrix create_rand_matrix(int r,int c){  // 建立亂數矩陣 
	srand((unsigned) time(NULL) + getpid());
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // 根據時間換值還要做 
			Data.data_matrix[r][c]=(double)rand()*2 / RAND_MAX + (-1);
		}
	}
	return Data;
}
Matrix create_one_matrix(int r,int c){ // 建立全部是1的矩陣
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // 根據時間換值還要做 
			Data.data_matrix[r][c]=1;
		}
	}
	return Data;
}
Matrix create_zero_matrix(int r,int c){ // 建立全部是0的矩陣
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // 根據時間換值還要做 
			Data.data_matrix[r][c]=0;
		}
	}
	return Data;
} 
Matrix matrix_tran_last_col_negative(Matrix Data){ //最後一行轉負值
	for(int i=0;i<Data.data_row;i++){
		Data.data_matrix[i][Data.data_col-1]=-1;
	}	
	return Data;
}
Matrix matrix_transpose(Matrix Matrix_1){  // 矩陣轉置 
	Matrix Matrix_tran=create_new_matrix(Matrix_1.data_col,Matrix_1.data_row);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_tran.data_matrix[j][i]=Matrix_1.data_matrix[i][j];
		}
	}	
	return Matrix_tran;
}
Matrix matrix_plus(Matrix Matrix_1,Matrix Matrix_2){ // 兩個矩陣大小相同，相加
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_1.data_col);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_sum.data_matrix[i][j]=Matrix_1.data_matrix[i][j]+Matrix_2.data_matrix[i][j];
		}
	}
	return Matrix_sum;
}
Matrix matrix_sub(Matrix Matrix_1,Matrix Matrix_2){ // 兩個矩陣大小相同，相減 
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_1.data_col);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_sum.data_matrix[i][j]=Matrix_1.data_matrix[i][j]-Matrix_2.data_matrix[i][j];
		}
	}
	return Matrix_sum;
}
Matrix matrix_mult(Matrix Matrix_1,Matrix Matrix_2){ // 兩個矩陣相乘 
 	Matrix_2=matrix_transpose(Matrix_2); // 第二個矩陣先轉置 
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_2.data_col);
	for(int i=0;i<Matrix_1.data_row;i++){ // 4 
		for(int j=0;j<Matrix_2.data_col;j++){ // 3
			double sum=0.0;
			for(int k=0;k<Matrix_2.data_row;k++){ // 2
				sum+=Matrix_1.data_matrix[i][k]*Matrix_2.data_matrix[k][j];
			}	
			Matrix_sum.data_matrix[i][j]=sum;
		}
	}
	return Matrix_sum;
}
Matrix matrix_hadamard(Matrix Matrix_1,Matrix Matrix_2){ // 哈達瑪積乘法矩陣，對應位置相乘，兩個矩陣大小相等
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_2.data_col);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_sum.data_matrix[i][j]=Matrix_1.data_matrix[i][j]*Matrix_2.data_matrix[i][j];
		}
	}
	return Matrix_sum;
}
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
double matrix_total_num(Matrix Data){ // 所有數值相加
	double total=0.0;
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			total+=Data.data_matrix[r][c];
		}
	}	
	return total;
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


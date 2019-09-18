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
void read_matrix_data(Matrix data){
	ifstream file("data.csv"); //讀入檔案  data_four  data  data_iris data_two
	for(int row=0;row<data.data_row;row++){
		string line;
		if(!getline(file,line))  //從輸入流讀入一行到string變量，直到沒有0讀入字符、返回false
			break;
		stringstream iss(line);  //將一個字符串string變量line的值轉成istringstream類別iss
		if(!iss.good())  //如果沒錯就回傳True
			break;
		for(int col=0;col<data.data_col;col++){
			string val;
			getline(iss,val,',');  //字串分割
			stringstream stringConvertorStringstream(val);  //將一個字符串變量的值傳遞給istringstream對象
			stringConvertorStringstream>>data.data_matrix[row][col];  //輸入到矩陣
		}
	}
	file.close();	
}
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
void printData(Matrix Data){ // 印出矩陣 
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			cout<<Data.data_matrix[r][c]<<",";
		}
		printf("\n");
	}printf("-----------\n");
}
Matrix re_zero(Matrix Data){ //歸零矩陣 
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data.data_matrix[r][c]=0.0;
		}
	}
	return Data;
}
Matrix create_rand_matrix(int r,int c){  // 建立亂數矩陣 
//	srand((unsigned) time(NULL) + getpid());
//(double) rand() / (RAND_MAX + 1.0 );
	/* 固定亂數種子 */
// 	srand(5);
 	srand( time(NULL) );
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // 根據時間換值
//			Data.data_matrix[r][c]=(double)rand()*2 / RAND_MAX + (-1);
			Data.data_matrix[r][c]=(1.0-0.2)*rand() / (RAND_MAX + 1.0) + (0.2);
		}
	}
	return Data;
}
Matrix create_one_matrix(int r,int c){ // 建立全部是1的矩陣
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // 根據時間換值還要做 
			Data.data_matrix[r][c]=1.0;
		}
	}
	return Data;
}
Matrix create_zero_matrix(int r,int c){ // 建立全部是0的矩陣
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // 根據時間換值還要做 
			Data.data_matrix[r][c]=0.0;
		}
	}
	return Data;
}
Matrix matrix_equal(Matrix Data){ // 使矩陣相等 
	Matrix Data_1=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_1.data_matrix[r][c]=Data.data_matrix[r][c];
		}
	}
	return Data_1;
}
Matrix matrix_tran_last_col_negative(Matrix Data){ //最後一行轉負值
	Matrix D=Data;
	for(int i=0;i<Data.data_row;i++){
		D.data_matrix[i][D.data_col-1]=-(D.data_matrix[i][D.data_col-1]);
	}	
	return D;
}
Matrix matrix_add_col_one(Matrix Data){ // 最後一行+1 
	Matrix Matrix_A=create_new_matrix(Data.data_row,Data.data_col+1);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Matrix_A.data_matrix[r][c]=Data.data_matrix[r][c];
		}
	}
	for(int i=0;i<Matrix_A.data_row;i++){
		Matrix_A.data_matrix[i][Matrix_A.data_col-1]=1.0;
	}	
	return Matrix_A;
}
Matrix matrix_delete_last_col_data(Matrix Data){  //刪除最後一直行 
	Matrix Matrix_A=create_new_matrix(Data.data_row,Data.data_col-1);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col-1;c++){
			Matrix_A.data_matrix[r][c]=Data.data_matrix[r][c];
		}
	}
	return Matrix_A;
}
Matrix matrix_get_one_row_data(Matrix Matrix_1,int row){  //取得某row data
	Matrix Data=create_new_matrix(1,Matrix_1.data_col);
	for(int c=0;c<Matrix_1.data_col;c++){ // 根據時間換值還要做 
		Data.data_matrix[0][c]=Matrix_1.data_matrix[row][c];
	}
	return Data;
}
Matrix matrix_row_sort_small_to_large(Matrix Data,int r){ // 小到大排序
	double tmp=0.0;
	for(int i=0;i<Data.data_col;i++){
		for(int j=0;j<Data.data_col-1;j++){
			if(Data.data_matrix[r][j]>Data.data_matrix[r][j+1]){
				tmp=Data.data_matrix[r][j];
				Data.data_matrix[r][j]=Data.data_matrix[r][j+1];
				Data.data_matrix[r][j+1]=tmp;
			}
		}
	}
	return Data;
}
Matrix matrix_get_col_label_data(Matrix Matrix_1,int c){ // 取得直行資料 欲知道種類個數(輸出層個數) 
	Matrix data_M=create_new_matrix(Matrix_1.data_row,Matrix_1.data_col);
	data_M=matrix_equal(Matrix_1); // 用一個新的矩陣來排序 
	for(int i=0;i<data_M.data_row;i++){	//----------找尋直行內的不重複屬性----------
 		for(int j=i+1;j<data_M.data_row;j++){
 			if(data_M.data_matrix[i][c]==data_M.data_matrix[j][c]){  //若第二個橫排跟第一個橫排一樣 
 				for(int k=j+1;k<data_M.data_row;k++){
 					data_M.data_matrix[k-1][c]=data_M.data_matrix[k][c]; //將第三個橫排往前放到第二個橫排 
				}
				--data_M.data_row; //行排數量減1 
				--j;
			}
		}
	}
	Matrix label=create_new_matrix(1,data_M.data_row);//二維矩陣的橫排排數轉成一維矩陣的col排數 
	//---------將找到的屬性作成陣列-----------
	for(int col=0;col<label.data_col;col++){
		label.data_matrix[0][col]=data_M.data_matrix[col][c]; // 矩陣成員
	}
	return label;	
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
 	Matrix_2=matrix_transpose(Matrix_2); // 矩陣2先轉置 
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_2.data_col); // 建立矩陣3 
	for(int i=0;i<Matrix_sum.data_row;i++){ // 矩陣3的row 
		for(int j=0;j<Matrix_sum.data_col;j++){ // 矩陣3的col
			double sum=0.0;
			for(int k=0;k<Matrix_2.data_row;k++){ // 矩陣2的row 
				sum+=Matrix_1.data_matrix[i][k]*Matrix_2.data_matrix[k][j];  
			}	
			Matrix_sum.data_matrix[i][j]=sum;
		}
	}
	return Matrix_sum;
}
Matrix matrix_mult_num(Matrix Matrix_1,double num){ // 矩陣乘上係數
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_1.data_matrix[i][j]*=num;
		}
	}	
	return Matrix_1;
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
double matrix_total(Matrix Data){ // 矩陣內所有數值相加
	double total=0.0;
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			total+=Data.data_matrix[r][c];
		}
	}	
	return total;
}


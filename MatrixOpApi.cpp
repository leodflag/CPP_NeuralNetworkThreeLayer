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
	ifstream file("data.csv"); //Ū�J�ɮ�  data_four  data  data_iris data_two
	for(int row=0;row<data.data_row;row++){
		string line;
		if(!getline(file,line))  //�q��J�yŪ�J�@���string�ܶq�A����S��0Ū�J�r�šB��^false
			break;
		stringstream iss(line);  //�N�@�Ӧr�Ŧ�string�ܶqline�����নistringstream���Oiss
		if(!iss.good())  //�p�G�S���N�^��True
			break;
		for(int col=0;col<data.data_col;col++){
			string val;
			getline(iss,val,',');  //�r�����
			stringstream stringConvertorStringstream(val);  //�N�@�Ӧr�Ŧ��ܶq���ȶǻ���istringstream��H
			stringConvertorStringstream>>data.data_matrix[row][col];  //��J��x�}
		}
	}
	file.close();	
}
Matrix create_new_matrix(int r,int c){ // �إ߰ʺA����
	Matrix Data;
	Data.data_row=r;
	Data.data_col=c;
	Data.data_matrix=NULL; //�ŧi�x�}
	Data.data_matrix=new double *[Data.data_row]; //�إߦ�data_row��string���}�C��}
	for(int i=0;i<Data.data_row;i++)
		Data.data_matrix[i]=new double[Data.data_col]; // �C���}�C��}���A�[data_col��string���}�C��}
	return Data;
}
void printData(Matrix Data){ // �L�X�x�} 
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			cout<<Data.data_matrix[r][c]<<",";
		}
		printf("\n");
	}printf("-----------\n");
}
Matrix re_zero(Matrix Data){ //�k�s�x�} 
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data.data_matrix[r][c]=0.0;
		}
	}
	return Data;
}
Matrix create_rand_matrix(int r,int c){  // �إ߶üƯx�} 
//	srand((unsigned) time(NULL) + getpid());
//(double) rand() / (RAND_MAX + 1.0 );
	/* �T�w�üƺؤl */
// 	srand(5);
 	srand( time(NULL) );
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // �ھڮɶ�����
//			Data.data_matrix[r][c]=(double)rand()*2 / RAND_MAX + (-1);
			Data.data_matrix[r][c]=(1.0-0.2)*rand() / (RAND_MAX + 1.0) + (0.2);
		}
	}
	return Data;
}
Matrix create_one_matrix(int r,int c){ // �إߥ����O1���x�}
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // �ھڮɶ������٭n�� 
			Data.data_matrix[r][c]=1.0;
		}
	}
	return Data;
}
Matrix create_zero_matrix(int r,int c){ // �إߥ����O0���x�}
	Matrix Data=create_new_matrix(r,c);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){ // �ھڮɶ������٭n�� 
			Data.data_matrix[r][c]=0.0;
		}
	}
	return Data;
}
Matrix matrix_equal(Matrix Data){ // �ϯx�}�۵� 
	Matrix Data_1=create_new_matrix(Data.data_row,Data.data_col);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			Data_1.data_matrix[r][c]=Data.data_matrix[r][c];
		}
	}
	return Data_1;
}
Matrix matrix_tran_last_col_negative(Matrix Data){ //�̫�@����t��
	Matrix D=Data;
	for(int i=0;i<Data.data_row;i++){
		D.data_matrix[i][D.data_col-1]=-(D.data_matrix[i][D.data_col-1]);
	}	
	return D;
}
Matrix matrix_add_col_one(Matrix Data){ // �̫�@��+1 
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
Matrix matrix_delete_last_col_data(Matrix Data){  //�R���̫�@���� 
	Matrix Matrix_A=create_new_matrix(Data.data_row,Data.data_col-1);
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col-1;c++){
			Matrix_A.data_matrix[r][c]=Data.data_matrix[r][c];
		}
	}
	return Matrix_A;
}
Matrix matrix_get_one_row_data(Matrix Matrix_1,int row){  //���o�Yrow data
	Matrix Data=create_new_matrix(1,Matrix_1.data_col);
	for(int c=0;c<Matrix_1.data_col;c++){ // �ھڮɶ������٭n�� 
		Data.data_matrix[0][c]=Matrix_1.data_matrix[row][c];
	}
	return Data;
}
Matrix matrix_row_sort_small_to_large(Matrix Data,int r){ // �p��j�Ƨ�
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
Matrix matrix_get_col_label_data(Matrix Matrix_1,int c){ // ���o������ �����D�����Ӽ�(��X�h�Ӽ�) 
	Matrix data_M=create_new_matrix(Matrix_1.data_row,Matrix_1.data_col);
	data_M=matrix_equal(Matrix_1); // �Τ@�ӷs���x�}�ӱƧ� 
	for(int i=0;i<data_M.data_row;i++){	//----------��M���椺���������ݩ�----------
 		for(int j=i+1;j<data_M.data_row;j++){
 			if(data_M.data_matrix[i][c]==data_M.data_matrix[j][c]){  //�Y�ĤG�Ӿ�Ƹ�Ĥ@�Ӿ�Ƥ@�� 
 				for(int k=j+1;k<data_M.data_row;k++){
 					data_M.data_matrix[k-1][c]=data_M.data_matrix[k][c]; //�N�ĤT�Ӿ�Ʃ��e���ĤG�Ӿ�� 
				}
				--data_M.data_row; //��Ƽƶq��1 
				--j;
			}
		}
	}
	Matrix label=create_new_matrix(1,data_M.data_row);//�G���x�}����ƱƼ��ন�@���x�}��col�Ƽ� 
	//---------�N��쪺�ݩʧ@���}�C-----------
	for(int col=0;col<label.data_col;col++){
		label.data_matrix[0][col]=data_M.data_matrix[col][c]; // �x�}����
	}
	return label;	
}
Matrix matrix_transpose(Matrix Matrix_1){  // �x�}��m 
	Matrix Matrix_tran=create_new_matrix(Matrix_1.data_col,Matrix_1.data_row);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_tran.data_matrix[j][i]=Matrix_1.data_matrix[i][j];
		}
	}	
	return Matrix_tran;
}
Matrix matrix_plus(Matrix Matrix_1,Matrix Matrix_2){ // ��ӯx�}�j�p�ۦP�A�ۥ[
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_1.data_col);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_sum.data_matrix[i][j]=Matrix_1.data_matrix[i][j]+Matrix_2.data_matrix[i][j];
		}
	}
	return Matrix_sum;
}
Matrix matrix_sub(Matrix Matrix_1,Matrix Matrix_2){ // ��ӯx�}�j�p�ۦP�A�۴� 
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_1.data_col);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_sum.data_matrix[i][j]=Matrix_1.data_matrix[i][j]-Matrix_2.data_matrix[i][j];
		}
	}
	return Matrix_sum;
}
Matrix matrix_mult(Matrix Matrix_1,Matrix Matrix_2){ // ��ӯx�}�ۭ� 
 	Matrix_2=matrix_transpose(Matrix_2); // �x�}2����m 
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_2.data_col); // �إ߯x�}3 
	for(int i=0;i<Matrix_sum.data_row;i++){ // �x�}3��row 
		for(int j=0;j<Matrix_sum.data_col;j++){ // �x�}3��col
			double sum=0.0;
			for(int k=0;k<Matrix_2.data_row;k++){ // �x�}2��row 
				sum+=Matrix_1.data_matrix[i][k]*Matrix_2.data_matrix[k][j];  
			}	
			Matrix_sum.data_matrix[i][j]=sum;
		}
	}
	return Matrix_sum;
}
Matrix matrix_mult_num(Matrix Matrix_1,double num){ // �x�}���W�Y��
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_1.data_matrix[i][j]*=num;
		}
	}	
	return Matrix_1;
} 
Matrix matrix_hadamard(Matrix Matrix_1,Matrix Matrix_2){ // ���F���n���k�x�}�A������m�ۭ��A��ӯx�}�j�p�۵�
	Matrix Matrix_sum=create_new_matrix(Matrix_1.data_row,Matrix_2.data_col);
	for(int i=0;i<Matrix_1.data_row;i++){
		for(int j=0;j<Matrix_1.data_col;j++){
			Matrix_sum.data_matrix[i][j]=Matrix_1.data_matrix[i][j]*Matrix_2.data_matrix[i][j];
		}
	}
	return Matrix_sum;
}
double matrix_total(Matrix Data){ // �x�}���Ҧ��ƭȬۥ[
	double total=0.0;
	for(int r=0;r<Data.data_row;r++){
		for(int c=0;c<Data.data_col;c++){
			total+=Data.data_matrix[r][c];
		}
	}	
	return total;
}


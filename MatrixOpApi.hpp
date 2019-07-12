#ifndef MATRIXOPAPI_HPP
#define MATRIXOPAPI_HPP
struct Matrix{
	double **data_matrix;
	int data_row;
	int data_col;
}; 
Matrix create_new_matrix(int r,int c); // 建立動態指標 OK 
void printData(Matrix Data); // 印出矩陣 OK 
Matrix creatMatrix(double A[][3],int r,int c); // 建立col=3的矩陣 
Matrix create_rand_matrix(int r,int c); // 建立亂數矩陣  OK 
Matrix create_one_matrix(int r,int c); // 建立全部是1的矩陣 OK 
Matrix create_zero_matrix(int r,int c); // 建立全部是0的矩陣 OK
Matrix matrix_tran_last_col_negative(Matrix Data); //最後一行轉負值 OK
Matrix matrix_transpose(Matrix Matrix_1);  // 矩陣轉置  OK
Matrix matrix_plus(Matrix Matrix_1,Matrix Matrix_2); // 兩個矩陣大小相同，相加 OK
Matrix matrix_sub(Matrix Matrix_1,Matrix Matrix_2); // 兩個矩陣大小相同，相減 OK
Matrix matrix_mult(Matrix Matrix_1,Matrix Matrix_2); // 兩個矩陣相乘 OK
Matrix matrix_hadamard(Matrix Matrix_1,Matrix Matrix_2); // 哈達瑪積乘法矩陣，對應位置相乘，兩個矩陣大小相等
Matrix matrix_sigmoid(Matrix Data); // 所有矩陣數值皆過sigmoid函數
Matrix matrix_sigmoid_der(Matrix Data); // 所有矩陣數值皆過sigmoid導函數
double matrix_total_num(Matrix Data);  // 所有數值相加
Matrix matrix_loss_function(Matrix Matrix_1,Matrix Matrix_2); // 損失函數( 目標矩陣，輸出矩陣 )
Matrix matrix_loss_function_der(Matrix Matrix_1,Matrix Matrix_2); // 損失導函數( 目標矩陣，輸出矩陣 )
#endif 


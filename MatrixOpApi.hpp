#ifndef MATRIXOPAPI_HPP
#define MATRIXOPAPI_HPP
struct Matrix{
	double **data_matrix;
	int data_row;
	int data_col;
}; 
void read_matrix_data(Matrix data); // 讀取csv檔的資料 
Matrix create_new_matrix(int r,int c); // 建立動態指標  
void printData(Matrix Data); // 印出矩陣 
Matrix re_zero(Matrix Data); // 歸零矩陣 
Matrix create_rand_matrix(int r,int c); // 建立亂數矩陣  
Matrix create_one_matrix(int r,int c); // 建立全部是1的矩陣 
Matrix create_zero_matrix(int r,int c); // 建立全部是0的矩陣 
Matrix matrix_equal(Matrix Data); // 使矩陣相等 
Matrix matrix_find_max(Matrix Data); // 找到每列最大的col位址 
Matrix matrix_tran_last_col_negative(Matrix Data); //最後一行轉負值 
Matrix matrix_add_col_one(Matrix Data); // 最後一行+1 
Matrix matrix_delete_last_col_data(Matrix Data);  //刪除最後一行 
Matrix matrix_get_one_row_data(Matrix Matrix_1,int row);  //取得某row data 
Matrix matrix_row_sort_small_to_large(Matrix Data,int r); // 小到大排序 
Matrix matrix_get_col_label_data(Matrix Matrix_1,int c); // 取得直行資料 欲知道種類個數(輸出層個數) 
Matrix matrix_transpose(Matrix Matrix_1);  // 矩陣轉置  
Matrix matrix_plus(Matrix Matrix_1,Matrix Matrix_2); // 兩個矩陣大小相同，相加 
Matrix matrix_sub(Matrix Matrix_1,Matrix Matrix_2); // 兩個矩陣大小相同，相減 
Matrix matrix_mult(Matrix Matrix_1,Matrix Matrix_2); // 兩個矩陣相乘 
Matrix matrix_mult_num(Matrix Matrix_1,double num); // 矩陣乘上係數 
Matrix matrix_hadamard(Matrix Matrix_1,Matrix Matrix_2); // 哈達瑪積乘法矩陣，對應位置相乘，兩個矩陣大小相等
double matrix_total(Matrix Data);  // 矩陣內所有數值相加
#endif 


#ifndef MATRIXOPAPI_HPP
#define MATRIXOPAPI_HPP
struct Matrix{
	double **data_matrix;
	int data_row;
	int data_col;
}; 
void read_matrix_data(Matrix data); // Ū��csv�ɪ���� 
Matrix create_new_matrix(int r,int c); // �إ߰ʺA���� OK 
void printData(Matrix Data); // �L�X�x�} OK 
Matrix create_rand_matrix(int r,int c); // �إ߶üƯx�}  OK 
Matrix create_one_matrix(int r,int c); // �إߥ����O1���x�} OK 
Matrix create_zero_matrix(int r,int c); // �إߥ����O0���x�} OK
Matrix matrix_tran_last_col_negative(Matrix Data); //�̫�@����t�� OK
Matrix matrix_delete_last_col_data(Matrix Matrix_1);  //�R���̫�@�� 
Matrix matrix_get_one_row_data(Matrix Matrix_1,int row);  //���o�Yrow data 
Matrix matrix_row_sort_small_to_large(Matrix Data,int r); // �p��j�Ƨ� 
Matrix matrix_get_col_lable_data(Matrix Matrix_1,int c); // ���o������ �����D�����Ӽ�(��X�h�Ӽ�) 
Matrix matrix_transpose(Matrix Matrix_1);  // �x�}��m  OK
Matrix matrix_plus(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�j�p�ۦP�A�ۥ[ OK
Matrix matrix_sub(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�j�p�ۦP�A�۴� OK
Matrix matrix_mult(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�ۭ� OK
Matrix matrix_mult_num(Matrix Matrix_1,double num); // �x�}���W�Y�� 
Matrix matrix_hadamard(Matrix Matrix_1,Matrix Matrix_2); // ���F���n���k�x�}�A������m�ۭ��A��ӯx�}�j�p�۵�

double matrix_total_num(Matrix Data);  // �Ҧ��ƭȬۥ[
#endif 


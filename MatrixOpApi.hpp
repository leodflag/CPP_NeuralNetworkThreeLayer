#ifndef MATRIXOPAPI_HPP
#define MATRIXOPAPI_HPP
struct Matrix{
	double **data_matrix;
	int data_row;
	int data_col;
}; 
void read_matrix_data(Matrix data); // Ū��csv�ɪ���� 
Matrix create_new_matrix(int r,int c); // �إ߰ʺA����  
void printData(Matrix Data); // �L�X�x�} 
Matrix re_zero(Matrix Data); // �k�s�x�} 
Matrix create_rand_matrix(int r,int c); // �إ߶üƯx�}  
Matrix create_one_matrix(int r,int c); // �إߥ����O1���x�} 
Matrix create_zero_matrix(int r,int c); // �إߥ����O0���x�} 
Matrix matrix_equal(Matrix Data); // �ϯx�}�۵� 
Matrix matrix_find_max(Matrix Data); // ���C�C�̤j��col��} 
Matrix matrix_tran_last_col_negative(Matrix Data); //�̫�@����t�� 
Matrix matrix_add_col_one(Matrix Data); // �̫�@��+1 
Matrix matrix_delete_last_col_data(Matrix Data);  //�R���̫�@�� 
Matrix matrix_get_one_row_data(Matrix Matrix_1,int row);  //���o�Yrow data 
Matrix matrix_row_sort_small_to_large(Matrix Data,int r); // �p��j�Ƨ� 
Matrix matrix_get_col_label_data(Matrix Matrix_1,int c); // ���o������ �����D�����Ӽ�(��X�h�Ӽ�) 
Matrix matrix_transpose(Matrix Matrix_1);  // �x�}��m  
Matrix matrix_plus(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�j�p�ۦP�A�ۥ[ 
Matrix matrix_sub(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�j�p�ۦP�A�۴� 
Matrix matrix_mult(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�ۭ� 
Matrix matrix_mult_num(Matrix Matrix_1,double num); // �x�}���W�Y�� 
Matrix matrix_hadamard(Matrix Matrix_1,Matrix Matrix_2); // ���F���n���k�x�}�A������m�ۭ��A��ӯx�}�j�p�۵�
double matrix_total(Matrix Data);  // �x�}���Ҧ��ƭȬۥ[
#endif 


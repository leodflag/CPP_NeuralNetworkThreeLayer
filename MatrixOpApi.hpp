#ifndef MATRIXOPAPI_HPP
#define MATRIXOPAPI_HPP
struct Matrix{
	double **data_matrix;
	int data_row;
	int data_col;
}; 
Matrix create_new_matrix(int r,int c); // �إ߰ʺA���� OK 
void printData(Matrix Data); // �L�X�x�} OK 
Matrix creatMatrix(double A[][3],int r,int c); // �إ�col=3���x�} 
Matrix create_rand_matrix(int r,int c); // �إ߶üƯx�}  OK 
Matrix create_one_matrix(int r,int c); // �إߥ����O1���x�} OK 
Matrix create_zero_matrix(int r,int c); // �إߥ����O0���x�} OK
Matrix matrix_tran_last_col_negative(Matrix Data); //�̫�@����t�� OK
Matrix matrix_transpose(Matrix Matrix_1);  // �x�}��m  OK
Matrix matrix_plus(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�j�p�ۦP�A�ۥ[ OK
Matrix matrix_sub(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�j�p�ۦP�A�۴� OK
Matrix matrix_mult(Matrix Matrix_1,Matrix Matrix_2); // ��ӯx�}�ۭ� OK
Matrix matrix_hadamard(Matrix Matrix_1,Matrix Matrix_2); // ���F���n���k�x�}�A������m�ۭ��A��ӯx�}�j�p�۵�
Matrix matrix_sigmoid(Matrix Data); // �Ҧ��x�}�ƭȬҹLsigmoid���
Matrix matrix_sigmoid_der(Matrix Data); // �Ҧ��x�}�ƭȬҹLsigmoid�ɨ��
double matrix_total_num(Matrix Data);  // �Ҧ��ƭȬۥ[
Matrix matrix_loss_function(Matrix Matrix_1,Matrix Matrix_2); // �l�����( �ؼЯx�}�A��X�x�} )
Matrix matrix_loss_function_der(Matrix Matrix_1,Matrix Matrix_2); // �l���ɨ��( �ؼЯx�}�A��X�x�} )
#endif 


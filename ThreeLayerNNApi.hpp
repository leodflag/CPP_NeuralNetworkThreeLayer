#ifndef THREELAYERNNAPI_HPP
#define THREELAYERNNAPI_HPP
#include "MatrixOpApi.hpp"
struct Net_layer{ // �����g�����h�A�x�}�Φ� 
	Matrix w;  // �v���x�}�A�tbais 
	Matrix delta_w; // �v���~�t 
	Matrix net; // ��J*�v���`�M row=>��ƭӼ�  col=>���g���Ӽ�
	Matrix net_sigmoid; // �w����  row=>��ƭӼ�  col=>���g���Ӽ�
	Matrix error; // �~�t�ȯx�}  row=>��ƭӼ�  col=>���g���Ӽ�
};
struct NeuralNetwork{  // �T�h���g���������c 
	Net_layer H_layer; // ���üh 
	Net_layer O_layer; // ��X�h 
};
Matrix matrix_sigmoid(Matrix Data); // �Ҧ��x�}�ƭȬҹLsigmoid���
Matrix matrix_sigmoid_der(Matrix Data); // �Ҧ��x�}�ƭȬҹLsigmoid�ɨ��
Matrix matrix_loss_function(Matrix Matrix_1,Matrix Matrix_2); // �l�����( �ؼЯx�}�A��X�x�} )
Matrix matrix_loss_function_der(Matrix Matrix_1,Matrix Matrix_2);// �l���ɨ��( �ؼЯx�}�A��X�x�} )
Matrix label_processing(Matrix Data);  // label �B�z�A���X���O���ҫ��one hot encoding�覡���s���� 
Matrix data_processing(Matrix Data);   // data �B�z�A�������O����A��J1�A�@��bais�ۭ��ɪ��v�� 
Matrix matrix_hidden_layer_error(Matrix weight,Matrix error); // �˶ǻ��ɭp�����üh���~��function 
Net_layer create_net_layer(int data_num,int col,int net_num); // �سy���g�h�A��J�`��ƾ��ӼơA����ӼơA���g���Ӽ�  
Matrix one_hot_encoding(Matrix data,Matrix label); // ���Ҥ覡�A���X���A��X�h�N���X�ӯ��g�� 
NeuralNetwork net_forward(NeuralNetwork NN,Matrix Data); // �e�V�Ǽ� 
NeuralNetwork net_back(NeuralNetwork NN,Matrix Label); //�˶ǻ� 
NeuralNetwork net_update_weight(NeuralNetwork NN,double learning_rate,Matrix Data); // �p��ç�s�v�� 
NeuralNetwork net_update_bais(NeuralNetwork NN,double learning_rate); // �p��ç�sbais 
NeuralNetwork BGD_calculate_delta_weight(NeuralNetwork NN,double learning_rate,Matrix Data); //�p��C���ƭȪ��v���Pbais�å[�_��
NeuralNetwork BGD_update_weight_and_bais(NeuralNetwork NN,int data_row); // ��s�v���Pbais�A�n�N�[�`�����~�����Ⱥ�X�ӡA���Hdata�� 
void save_weight(NeuralNetwork NN); // �x�s���üh�B��X�h�v�� 
void save_nn_structure(NeuralNetwork NN,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration); // �x�s���g�����[�c 
void printALLData(NeuralNetwork NN); // �L�X���g�������üh�B��X�h���ƭ� 
void SGD(Matrix Data,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration,double stop_err); // �H����פU�� 
void BGD(Matrix Data,int hidden_net_num,int output_net_num,int data_col,double learning_rate,int iteration); // �H����פU�� 
#endif

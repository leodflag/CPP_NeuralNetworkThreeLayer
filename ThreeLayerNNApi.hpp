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
struct NeuralNetwork{
	Net_layer H_layer;
	Net_layer O_layer;
};
Matrix matrix_sigmoid(Matrix Data); // �Ҧ��x�}�ƭȬҹLsigmoid���
Matrix matrix_sigmoid_der(Matrix Data); // �Ҧ��x�}�ƭȬҹLsigmoid�ɨ��
Matrix matrix_loss_function(Matrix Matrix_1,Matrix Matrix_2); // �l�����( �ؼЯx�}�A��X�x�} )
Matrix matrix_loss_function_der(Matrix Matrix_1,Matrix Matrix_2);// �l���ɨ��( �ؼЯx�}�A��X�x�} )
Matrix label_processing(Matrix Data); 
Matrix data_processing(Matrix Data);
Matrix matrix_hidden_layer_error(Matrix weight,Matrix error);
Net_layer create_net_layer(int data_num,int col,int net_num); // �سy���g�h�A��J�`��ƾ��ӼơA����ӼơA���g���Ӽ�  
Matrix one_hot_encoding(Matrix data,Matrix label); // ������ 
NeuralNetwork net_forward(NeuralNetwork NN,Matrix Data); // �e�V�Ǽ� 
NeuralNetwork net_back(NeuralNetwork NN,Matrix Label); //�˶ǻ� 
NeuralNetwork net_update_weight(NeuralNetwork NN,double learning_rate,Matrix Data);
NeuralNetwork net_update_bais(NeuralNetwork NN,double learning_rate);
void printALLData(NeuralNetwork NN);
#endif

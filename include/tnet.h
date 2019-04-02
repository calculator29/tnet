// �o�[�W����	�F 4.0
// �쐬����	�F 2015/11/21

#pragma once

// C �����^�C�� �w�b�_�[ �t�@�C��
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <stdarg.h>
#include <time.h>
#include <fcntl.h>

// C++ �t�@�C��
#include <iostream>
#include <fstream>
#include <vector>

// Eigen �t�@�C��
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

/* �p�[�Z�v�g���� */
class NeuralNet {
 private:
  double T = 1.0;
  double alfa = 0.1;	// �d�݂̊w�K��
  double beta = 0.1;	// 臒l�̊w�K��
  double reg = 1E-4;	// �����̏d��

 public:
  vector<int> Size;

  vector<RowVectorXd*> Y;	//���ꂼ��̏o�͒l
  vector<RowVectorXd*> Delta;	//�덷�`�d�p
  vector<RowVectorXd*> Theta;	//臒l
  vector<MatrixXd*> Wait;		//���ꂼ��̏d��
  vector<MatrixXd*> dWait;


  /* �R���X�g���N�^ */
  /* ���C���[���C���͑w�C�B��w�C... �C�o�͑w */
  NeuralNet(const int layer, ...);

  /* �f�X�g���N�^ */
  ~NeuralNet();

  /* �V�O���C�h�֐� */
  RowVectorXd Sigmoid(const RowVectorXd &x);

  /* �V�O���C�h�̔��� */
  RowVectorXd d_Sigmoid(const RowVectorXd &y);

  /* ���`�֐� */
  RowVectorXd Liner(const RowVectorXd &x);

  /* ���`�֐����� */
  RowVectorXd d_Liner(const RowVectorXd &y);

  /* �o�� */
  RowVectorXd operator()(const RowVectorXd &Input);

  /* �w�K */
  void Study(const RowVectorXd &Error);

  void ChangeAlfa(double value) {
    alfa = value;
    beta = value;
  }
};

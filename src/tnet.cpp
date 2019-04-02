#include "tnet.h"

#define drandom(min, max) ((double)rand() / RAND_MAX * (max - min) + min)

/* コンストラクタ */
NeuralNet::NeuralNet(const int layer, ...) {

  /* 各レイヤーのニューロン数を保存 */
  va_list args;
  va_start(args, layer);
  for (int i = 0; i < layer; i++) {
    Size.push_back(va_arg(args, int));
  }
  va_end(args);

  /* 出力値 */
  for (int i = 0; i < layer; i++) {
    Y.push_back(new RowVectorXd(Size[i]));
  }

  /* 誤差伝播用 */
  for (int i = 0; i < layer - 1; i++) {
    Delta.push_back(new RowVectorXd(Size[i + 1]));
  }

  /* 閾値用 */
  for (int i = 0; i < layer - 1; i++) {
    Theta.push_back(new RowVectorXd(Size[i + 1]));
    (*Theta.back()) = RowVectorXd::Random(Size[i + 1]) * 0.0; //初期化
  }

  /* 重みの設定 */
  for (int i = 0; i < layer - 1; i++) {
    Wait.push_back(new MatrixXd(Size[i], Size[i + 1]));
    (*Wait.back()) = MatrixXd::Random(Size[i], Size[i + 1]) * 3.0; //初期化
  }

  for (int i = 0; i < layer - 1; i++) {
    dWait.push_back(new MatrixXd(Size[i], Size[i + 1]));
    (*dWait.back()) = MatrixXd::Random(Size[i], Size[i + 1]) * 3.0; //初期化
  }
}

/* デストラクタ */
NeuralNet::~NeuralNet() {
  /* Y開放 */
  while (Y.size() > 0) {
    delete Y.back();
    Y.pop_back();
  }

  /* Delta開放 */
  while (Delta.size() > 0) {
    delete Delta.back();
    Delta.pop_back();
  }

  /* Theta開放 */
  while (Theta.size() > 0) {
    delete Theta.back();
    Theta.pop_back();
  }

  /* Wait開放 */
  while (Wait.size() > 0) {
    delete Wait.back();
    Wait.pop_back();
  }

  while (dWait.size() > 0) {
    delete dWait.back();
    dWait.pop_back();
  }
}

/* シグモイド関数 */
RowVectorXd NeuralNet::Sigmoid(const RowVectorXd &x) {
  return 1.0 / (1.0 + (-1 * x / T).array().exp());
}

/* シグモイドの微分 */
RowVectorXd NeuralNet::d_Sigmoid(const RowVectorXd &y) {
  return y.array() * (1.0 - y.array()) / T;
}

/* 線形関数 */
RowVectorXd NeuralNet::Liner(const RowVectorXd &x) {
  return x;
}

/* 線形関数微分 */
RowVectorXd NeuralNet::d_Liner(const RowVectorXd &y) {
  return RowVectorXd::Ones(y.size());
}

/* 出力 */
RowVectorXd NeuralNet::operator()(const RowVectorXd &Input) {
  (*Y[0]) = Input;	//入力層はそのまま
  for (int i = 0; i < Y.size() - 1; i++) {
    if (i + 1 == Y.size() - 1)	(*Y[i + 1]) = Liner((*Y[i]) * (*Wait[i]) + (*Theta[i]));
    else					(*Y[i + 1]) = Sigmoid((*Y[i]) * (*Wait[i]) + (*Theta[i]));
  }
  return (*Y.back());
}

/* 学習 */
void NeuralNet::Study(const RowVectorXd &Error) {
  /* 誤差伝播 */
  (*Delta.back()) = -1.0 * Error;
  for (int i = Delta.size() - 1; i > 0; i--) {
    if (i + 1 == Y.size() - 1)	(*Delta[i - 1]) = (*Delta[i]).cwiseProduct(d_Liner((*Y[i + 1]))) * (*Wait[i]).transpose();
    else						(*Delta[i - 1]) = (*Delta[i]).cwiseProduct(d_Sigmoid((*Y[i + 1]))) * (*Wait[i]).transpose();
  }

  /* 勾配法 */
  for (int i = 0; i < Delta.size() - 1; i++) {
    if (i + 1 == Y.size() - 1) {
      (*Wait[i]) -= (*dWait[i]) = alfa * ((*Y[i]).transpose() * (*Delta[i]).cwiseProduct(d_Liner((*Y[i + 1])))) + alfa * reg * (*Wait[i]).cwiseQuotient((*Wait[i]).cwiseAbs());
      (*Theta[i]) -= beta * (*Delta[i]).cwiseProduct(d_Liner((*Y[i + 1])));
    }
    else {

      (*Wait[i]) -= (*dWait[i]) = alfa * ((*Y[i]).transpose() * (*Delta[i]).cwiseProduct(d_Sigmoid((*Y[i + 1])))) + alfa * reg * (*Wait[i]).cwiseQuotient((*Wait[i]).cwiseAbs());
      (*Theta[i]) -= beta * (*Delta[i]).cwiseProduct(d_Sigmoid((*Y[i + 1])));
    }
  }
}

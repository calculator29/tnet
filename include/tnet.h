// バージョン	： 4.0
// 作成日時	： 2015/11/21

#pragma once

// C ランタイム ヘッダー ファイル
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <stdarg.h>
#include <time.h>
#include <fcntl.h>

// C++ ファイル
#include <iostream>
#include <fstream>
#include <vector>

// Eigen ファイル
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

/* パーセプトロン */
class NeuralNet {
 private:
  double T = 1.0;
  double alfa = 0.1;	// 重みの学習率
  double beta = 0.1;	// 閾値の学習率
  double reg = 1E-4;	// 正則の重み

 public:
  vector<int> Size;

  vector<RowVectorXd*> Y;	//それぞれの出力値
  vector<RowVectorXd*> Delta;	//誤差伝播用
  vector<RowVectorXd*> Theta;	//閾値
  vector<MatrixXd*> Wait;		//それぞれの重み
  vector<MatrixXd*> dWait;


  /* コンストラクタ */
  /* レイヤー数，入力層，隠れ層，... ，出力層 */
  NeuralNet(const int layer, ...);

  /* デストラクタ */
  ~NeuralNet();

  /* シグモイド関数 */
  RowVectorXd Sigmoid(const RowVectorXd &x);

  /* シグモイドの微分 */
  RowVectorXd d_Sigmoid(const RowVectorXd &y);

  /* 線形関数 */
  RowVectorXd Liner(const RowVectorXd &x);

  /* 線形関数微分 */
  RowVectorXd d_Liner(const RowVectorXd &y);

  /* 出力 */
  RowVectorXd operator()(const RowVectorXd &Input);

  /* 学習 */
  void Study(const RowVectorXd &Error);

  void ChangeAlfa(double value) {
    alfa = value;
    beta = value;
  }
};

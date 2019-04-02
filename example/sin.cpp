#include <tnet.h>

constexpr double PI=3.141592;
using namespace Eigen;

int main(void){
  NeuralNet net(3, 1, 10, 1);
  VectorXd in(1), err(1);
  for(int i=0;i<1000;i++){
    for(double theta=-PI;theta<PI;theta+=0.1){
      in[0] = theta;
      err[0] = sin(in[0]) - net(in)[0];
      net.Study(err);
    }
  }

  for(double theta=-PI;theta<PI;theta+=0.01){
    in[0] = theta;
    cout << theta << " " << net(in)[0] << endl;
  }


  return 0;
}

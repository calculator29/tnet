#include <tnet.h>

using namespace Eigen;

constexpr double PI=3.141592;
constexpr double dt=0.01;
double K=10, C=1;

double control(double x, double v){
  return -K*x-C*v;
}

void update(double &x, double &v, double u){
  double ext = 2;
  v+=(u+ext)*dt;
  x+=v*dt;
}

int main(int argc, char* argv[]){
  NeuralNet net(3, 3, 50, 1);
  VectorXd in(3), out(1), err(1);
  bool m=true;
  if(argc>1) if(argv[1][0]=='b') m=false;

  if(m){
    for(int i=0;i<1000;i++){
      cerr<<i<<endl;
      double w=(double)i/1000.0;
      double t=0;
      double x=2.0*(double)rand()/(double)RAND_MAX-1.0, v=0;
      K=10*(double)rand()/(double)RAND_MAX;
      C=10*(double)rand()/(double)RAND_MAX;
      while(t<2.0){
        t+=dt;
        in[0]=0;in[1]=x;in[2]=v;
        out=net(in);

        double u_fb = (1-w)*control(x,v);
        double u_ff = w*out[0];
        double u = u_fb + u_ff;
        update(x,v,u);

        err[0] = u_fb;
        net.Study(err);
      }
    }
  }

  double w=(m?1:0);
  double t=0;
  double x=1, v=0;
  K=10.0;
  C=1.0;
  while(t<5.0){
    t+=dt;
    in[0]=0;in[1]=x;in[2]=v;
    out=net(in);

    double u_fb = (1-w)*control(x,v);
    double u_ff = w*out[0];
    double u = u_fb + u_ff;
    update(x,v,u);

    cout<<t<<" "<<x<<endl;
  }

  return 0;
}

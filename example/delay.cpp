#include <tnet.h>

using namespace Eigen;

constexpr double PI=3.141592;
constexpr double dt=0.01;


double Kp=10, Kd=1;
double control(double x, double v){
  return -Kp*x-Kd*v;
}

double K=100, C=20;
double servo(double x, double v, double dx){
  return K*(dx-x)-C*v;
}

void update(double &x, double &v, double u){
  v+=u*dt;
  x+=v*dt;
}

double feedback(double a, double da){
  return 20*(da-a);
}



int main(int argc, char* argv[]){
  NeuralNet net(3, 6, 100, 1);
  VectorXd in(6), out(1), err(1);
  bool m=true;
  if(argc>1) if(argv[1][0]=='b') m=false;

  if(m){
    int N=1000;
    for(int i=0;i<N;i++){
      cerr<<i<<endl;
      double w=(double)i/(double)N;
      double t=0;
      double a=0, x=2.0*(double)rand()/(double)RAND_MAX-1.0, v=0;
      double mx=x, mv=v;
      while(t<2.0){
        t+=dt;
        double da = control(mx,mv);

        in[0]=da;in[1]=a;in[2]=x,in[3]=v;in[4]=mx,in[5]=mv;
        out=net(in);

        double u_fb = (1-w)*feedback(a,da);
        double u_ff = w*out[0];
        double u = u_fb + u_ff;

        update(mx,mv,da+u);
        a = servo(x,v,mx);
        update(x,v,a);
        mx+=dt*(x-mx); mv=dt*(v-mv);

        err[0] = u_fb;
        net.Study(err);
      }
    }
  }

  double w=(m?1:0);
  double t=0;
  double a=0, x=1, v=0;
  double mx=x, mv=v;
  while(t<5.0){
    t+=dt;
    double da = control(mx,mv);

    in[0]=da;in[1]=a;in[2]=x,in[3]=v;in[4]=mx,in[5]=mv;
    out=net(in);

    double u_fb = (1-w)*feedback(a,da);
    double u_ff = w*out[0];
    double u = u_fb + u_ff;

    update(mx,mv,da+u);
    a = servo(x,v,mx);
    update(x,v,a);
    mx+=dt*(x-mx); mv=dt*(v-mv);

    cout<<t<<" "<<x<<endl;
  }

  return 0;
}

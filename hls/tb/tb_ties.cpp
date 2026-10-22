// Does the segmented minimum match the full scan on TIE-HEAVY data?
// Quantised frontiers collide constantly -- FP4 has seven magnitudes --
// so ties are the common case, not the corner case.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "spmm_accel.h"
static void ref_insert(float*v,int*c,int&n,float&mv,int&mi,int k,float x,int col){
  float a=fabsf(x); if(a<1e-12f) return;
  if(n<k){ v[n]=x;c[n]=col; if(n==0||a<mv){mv=a;mi=n;} n++; }
  else { if(a<=mv) return; v[mi]=x;c[mi]=col;
    float nm=fabsf(v[0]); int ni=0;
    for(int i=1;i<k;i++){float q=fabsf(v[i]); if(q<nm){nm=q;ni=i;}}
    mv=nm;mi=ni; } }
static void seg_insert(float*v,int*c,int&n,float&mv,int&mi,float*sm,int*si,
                       int k,float x,int col){
  float a=fabsf(x); if(a<1e-12f) return;
  if(n<k){ int p=n; v[p]=x;c[p]=col; int s=p/TK_SEG;
    if(a<sm[s]){sm[s]=a;si[s]=p;} if(n==0||a<mv){mv=a;mi=p;} n++; }
  else { if(a<=mv) return; int p=mi; v[p]=x;c[p]=col;
    int s=p/TK_SEG,b=s*TK_SEG; float m2=3.4e38f; int i2=b;
    for(int t=0;t<TK_SEG;t++){int q=b+t; if(q<k){float qa=fabsf(v[q]); if(qa<m2){m2=qa;i2=q;}}}
    sm[s]=m2; si[s]=i2;
    float g=3.4e38f; int gi=0;
    for(int t=0;t<TK_NSEG;t++) if(t*TK_SEG<k && sm[t]<g){g=sm[t];gi=si[t];}
    mv=g;mi=gi; } }
int main(){
  srand(20261019);
  for (int levels : {7, 16, 64, 0}) {   // 0 = continuous
    int bad=0,trials=0,col_bad=0; double sum_a=0,sum_b=0;
    for(int t=0;t<80;t++){
      int k=200,N=k*8;
      std::vector<float> rv(k),sv(k); std::vector<int> rc(k),sc(k);
      std::vector<float> sm(TK_NSEG,3.4e38f); std::vector<int> si(TK_NSEG,0);
      int rn=0,sn=0,rmi=0,smi=0; float rmv=0,smv=0;
      for(int i=0;i<N;i++){
        float x;
        if(levels) x = powf(2.0f, -(float)(rand()%levels));  // few distinct magnitudes
        else       x = powf(10.0f, -4.0f*(float)rand()/RAND_MAX);
        int col=rand()%100000;
        ref_insert(rv.data(),rc.data(),rn,rmv,rmi,k,x,col);
        seg_insert(sv.data(),sc.data(),sn,smv,smi,sm.data(),si.data(),k,x,col);
      }
      // compare the multiset of retained MAGNITUDES (what top-k means)
      std::vector<float> a(rv.begin(),rv.begin()+rn), b(sv.begin(),sv.begin()+sn);
      std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
      { std::vector<int> ca(rc.begin(),rc.begin()+rn), cb(sc.begin(),sc.begin()+sn);
        std::sort(ca.begin(),ca.end()); std::sort(cb.begin(),cb.end());
        if (ca!=cb) col_bad++; }
      double sa=0,sb=0; for(float z:a) sa+=fabs(z); for(float z:b) sb+=fabs(z);
      sum_a+=sa; sum_b+=sb;
      trials++; if(a!=b) bad++;
    }
    printf("%-22s magnitude-multiset identical: %3d/%3d   retained mass ref=%.6g seg=%.6g (%+.3f%%)\n",
           levels? "quantised levels" : "continuous",
           trials-bad, trials, sum_a, sum_b, 100.0*(sum_b-sum_a)/sum_a);
    printf("%-22s   column-set identical: %3d/%3d\n", "", trials-col_bad, trials);
  }
  return 0; }

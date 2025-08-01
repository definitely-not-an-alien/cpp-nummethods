#include "GradDesc.h"
#include <bits/stdc++.h>
using namespace std;

template <typename T> void printMat(numMethods::Matrix<T> &x){
    for(int i=0;i<x.getRows();i++){
        for(int j=0;j<x.getCols();j++){
            cout<<x[i][j]<<" ";
        }
        cout<<"\n";
    }
}
template <typename T> void printVec(numMethods::NumVector<T> &x){
    for(int i=0;i<x.getSize();i++){
        cout<<x[i]<<" ";
    }
    cout<<"\n";
}
int gcdc(int a, int b){
    if(b==0) return a;
    else return gcdc(b, a%b);
}
int fact(int a, int b){
    return a / gcdc(a, b);
}
float divc(float a, float b){
    return a/b;
}
float cons(float a, float b){
    return 1.0;
}
int main(){
    float arr[5] = {1,2,3,4,5}, arr2[5] = {2,3,4,5,6};
    numMethods::NumVector<float> ve(5, arr), ve2(5,arr2);
    cout<<"Hello World!\n";
    numMethods::NumVector<float> ve3 = ve+ve2;
    cout<<ve.dot(ve2)<<"\n";
    float* res = ve3.getNums();
    cout<<"ve3: ";
    for(float* i = res; i != res + 5; i++){
        cout<<(*i)<<" ";
    }
    cout<<"\n\n";
    cout<<"\n";
    ve3 += ve2;
    ve3 = (float)5.0 * ve3;
    res = ve3.getNums();
    cout<<"ve3: ";
    for(int i = 0; i < 5; i++){
        cout<<ve3[i]<<" ";
    }
    cout<<"\n";
    cout<<ve3.norm()<<"\n";
    ve3[3] = 1;
    for(int i = 0; i < 5; i++){
        cout<<ve3[i]<<" ";
    }
    cout<<"\n";
    cout<<ve3.norm()<<"\n";
    numMethods::NumVector<float> ve4 = ve3.normalise();
    for(int i = 1; i <= 5; i++){
        cout<<ve4.ele(i)<<" ";
    }
    cout<<"\n";
    cout<<ve4.norm()<<"\n";
    ve4.ele(2)+=1;
    for(int i = 1; i <= 5; i++){
        cout<<ve4.ele(i)<<" ";
    }
    cout<<"\n";
    cout<<"\n";
    ve4 -= ve3;
    for(int i = 1; i <= 5; i++){
        cout<<ve4.ele(i)<<" ";
    }
    cout<<"\n";
    cout<<ve4.leading().index<<" "<<ve4.leading().value<<"**\n";
    int test[2][3] = {{1, 2, 3},{4, 5, 6}};
    /*
    1 2 3
    4 5 6
     */
    numMethods::Matrix<int> m1(2,3,(int*)test);
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            cout<<m1[i][j]<<" ";
        }
        cout<<"\n";
    }
    cout<<m1.getRows()<<" "<<m1.getCols()<<"\n";
    numMethods::Matrix<int> m2 = m1.transposed();
    m1.transpose();
    cout<<m2.getRows()<<" "<<m2.getCols()<<"\n";
    m2 = 10 * m2;
    for(int i=0;i<3;i++){
        for(int j=0;j<2;j++){
            cout<<m2[i][j]<<" ";
        }
        cout<<"\n";
    }
    cout<<"\n";
    m2.cswap(1,0);
    for(int i=0;i<3;i++){
        for(int j=0;j<2;j++){
            cout<<m2[i][j]<<" ";
        }
        cout<<"\n";
    }
    cout<<"\n";
    int* col = m2.col(0);
    for(int j=0;j<3;j++){
        cout<<col[j]<<"\n";
    }
    cout<<"\n";
    int* row = m1.row(1);
    for(int j=0;j<2;j++){
        cout<<row[j]<<" ";
    }
    cout<<"\n";
    m1.transpose();
    printMat(m1);
    printMat(m2);
    numMethods::Matrix<int> m3 = m1*m2;
    m1 *= m2;
    m2.transpose();
    m2 *= m2.transposed();
    printMat(m3);
    numMethods::Matrix<int> m4 = m2*m1;
    printMat(m4);
    printMat(m1);
    float mat[4][5] = {
        {1, 3, 3, 8, 5},
        {0, 1, 3, 10, 8},
        {0, 0, 0, -1, -4},
        {0, 0, 0, 2, 8}
    };
    int mat2[3][3] = {
        {2, 0, 0},
        {1, 0, 1},
        {2, 5, 3}
    };
    numMethods::Matrix<float> m5(4,5,(float*)mat);
    // m5.set(0,0,2);
    // m5.set(1,0,1);
    // m5.set(1,1,1);
    m1 = *(new numMethods::Matrix<int>(3,3,(int*)mat2));
    m3 = m1.echRed(fact,fact);
    printMat(m3);
    // m2 = m1.echRedNoPivot(fact,fact);
    // printMat(m2);
    // numMethods::Matrix<float> ret(m1.getRows(),m1.getCols(), (float*)m1.getNums(0),(float*)m1.getNums(1));
    // printMat(ret);
    m5 = m5.RREF();
    printMat(m5);
    m5 = m1.RREF();
    printMat(m5);
    numMethods::NumVector<int> r=numMethods::UnitVect<int>(5,1);
    printVec(r);
    int mat3[2][2] = {
        {2, 1},
        {1, 0}
    };
    m1 = *(new numMethods::Matrix<int>(3,3,(int*)mat2));
    printMat(m1);
    cout<<"\n";
    cout<<"det(m1): "<<m1.determinant()<<"\n";
    cout<<"___m1____________\n";
    printMat(m1);
    cout<<"___m1^(-1)_______\n";
    m5 = m1.inverse();
    printMat(m5);
    cout<<"___m1 * (m1^-1)__\n";
    m5 *= m1.convert();
    printMat(m5);
}
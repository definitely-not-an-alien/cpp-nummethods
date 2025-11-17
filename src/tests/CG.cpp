#include "NumMethodsCpp.h"
#include <bits/stdc++.h>
using namespace std;
using namespace numMethods;
template <typename T> void printMat(numMethods::Matrix<T> &x){
    for(int i=0;i<x.getRows();i++){
        for(int j=0;j<x.getCols();j++){
            cout<<x[i][j]<<" ";
        }
        cout<<"\n";
    }
}



int main(){

}
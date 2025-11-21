#include <cassert>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <NumMatrix.h>
#include <NumVector.h>

#ifndef __MATMATRIX_H__
#define __MATMATRIX_H__
namespace numMethods {
// Matrix class for any number type
template <typename T> class Matrix;
// Standard matrices
// Zero matrix
template <typename T> Matrix<T> ZeroMat(size_t dim);
// Identity matrix
template <typename T> Matrix<T> Iden(size_t dim);

// Vector class for any number type, assumed to be column vector
template <typename T> class NumVector;
// Standard vectors
// Zero vector
template <typename T2> NumVector<T2> ZeroVect(size_t sz);
// en base vector
template <typename T2> NumVector<T2> UnitVect(size_t sz, uint32_t n);
namespace MatMethods {


// Pivot, Lower-triangular, Upper-triangular
struct PLU{
    Matrix<float> P, L, U;
};
// PLU Factorisation (pivoting only prevents division by 0)
PLU PLUfactorize(Matrix<float>& __this) {
    assert(__this.getRows()==__this.getCols());
    uint32_t __rows = __this.getRows();
    Matrix<float> P = Iden<float>(__rows), L = Iden<float>(__rows), U=__this.convert();
    int __i,__j,__k,currLead=0;
    float leadVal=0;
    __k = 0;
    for(__i=0;__i<__rows;__i++){
        float maximum = 0;
        int ind = __rows;
        for(__j=__k;__j<__rows;__j++){
            if(U[__j][__i]){
                if(abs(U[__j][__i])>maximum){
                    ind=__j;
                    maximum=abs(U[__j][__i]);
                }
            }
        }
        assert(ind!=__rows);
        if(ind!=__k){
            U.rswap(__k,ind);
            P.rswap(__k,ind);
        }
        NumVector<float>curr = U.extractNumRow(__k);
        assert(curr.leading().index>=currLead);
        currLead = curr.leading().index;
        leadVal = curr.leading().value;
        if(leadVal){
            for(__j=__k+1;__j<__rows;__j++){
                NumVector<float>target = U.extractNumRow(__j);
                float targLead=target[currLead];
                if(targLead==0)continue;
                float uValCurr = targLead/leadVal;
                target -= uValCurr * curr;
                target.adjust();
                U.setRow(__j,target);
                L.set(__j,__k,uValCurr);
            }
        }
        __k++;
    }
    PLU ret;
    ret.P = P; 
    ret.L = L;  
    ret.U = U;  
    return ret;
}
PLU PLUfactorize(const Matrix<float>& __this) {
    assert(__this.getRows()==__this.getCols());
    uint32_t __rows = __this.getRows();
    Matrix<float> P = Iden<float>(__rows), L = Iden<float>(__rows), U=__this.convert();
    int __i,__j,__k,currLead=0;
    float leadVal=0;
    __k = 0;
    for(__i=0;__i<__rows;__i++){
        float maximum = 0;
        int ind = __rows;
        for(__j=__k;__j<__rows;__j++){
            if(U[__j][__i]){
                if(abs(U[__j][__i])>maximum){
                    ind=__j;
                    maximum=abs(U[__j][__i]);
                }
            }
        }
        assert(ind!=__rows);
        if(ind!=__k){
            U.rswap(__k,ind);
            P.rswap(__k,ind);
        }
        NumVector<float>curr = U.extractNumRow(__k);
        assert(curr.leading().index>=currLead);
        currLead = curr.leading().index;
        leadVal = curr.leading().value;
        if(leadVal){
            for(__j=__k+1;__j<__rows;__j++){
                NumVector<float>target = U.extractNumRow(__j);
                float targLead=target[currLead];
                if(targLead==0)continue;
                float uValCurr = targLead/leadVal;
                target -= uValCurr * curr;
                target.adjust();
                U.setRow(__j,target);
                L.set(__j,__k,uValCurr);
            }
        }
        __k++;
    }
    PLU ret;
    ret.P = P; 
    ret.L = L;  
    ret.U = U;  
    return ret;
}
}
}
#endif
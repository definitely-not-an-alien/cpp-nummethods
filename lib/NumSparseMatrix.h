#include <cassert>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>

#ifndef __NUMSPARSEMATRIX_H__
#define __NUMSPARSEMATRIX_H__
namespace numMethods {
// Sparse matrix class for memory / time efficiency, limited functions implemented to ensure sparseness, no factorisations implemented (convert back to Matrix before factorisation)
template <typename T> class SparseMatrix {
    protected:
        size_t rows = 0, cols = 0, elements = 0;
        T* entries;
        uint32_t rpos, cpos;
    public:
        // Constructors
        SparseMatrix(){
        }
        // Set everything directly
        SparseMatrix<T>(uint32_t r, uint32_t c, T* ent, uint32_t* rp, uint32_t* cp){
            rows = r;
            cols = c;
            elements = r * c;
            entries = (T*)malloc(elements*sizeof(T));
            memcpy(entries,ent,elements*sizeof(T));
            rpos = (uint32_t*)malloc(elements*sizeof(uint32_t));
            memcpy(rpos,rp,elements*sizeof(uint32_t));
            cpos = (uint32_t*)malloc(elements*sizeof(uint32_t));
            memcpy(cpos,cp,elements*sizeof(uint32_t));
        }
        // Build from 2D array
        SparseMatrix<T>(uint32_t r, uint32_t c, T* ent){
            rows = r;
            cols = c;
            elements = 0;
            for(uint32_t __i=0;__i<r;__i++){
                for(uint32_t __j=0;__j<c;__j++){
                    elements+=(ent[__i*c+__j]!=0);
                }
            }
            entries = (T*)malloc(elements*sizeof(T));
            rpos = (uint32_t*)malloc(elements*sizeof(uint32_t));
            cpos = (uint32_t*)malloc(elements*sizeof(uint32_t));
            uint32_t __k = 0;
            for(uint32_t __i=0;__i<r;__i++){
                for(uint32_t __j=0;__j<c;__j++){
                    if(ent[__i*c+__j]){entries[__k]=ent[__i*c+__j];rpos[__k]=__i;cpos[__k]=__j;__k++;}
                }
            }
        }
        // Matrix scaling
        SparseMatrix<T> operator*(T const& factor) const{
            T* temp = (T*)malloc(elements*sizeof(T));
            uint32_t* rtemp = (uint32_t*)malloc(elements*sizeof(uint32_t));
            uint32_t* ctemp = (uint32_t*)malloc(elements*sizeof(uint32_t));
            memcpy(rtemp,rpos,sizeof(rtemp));
            memcpy(ctemp,cpos,sizeof(ctemp));
            for(int __i=0;__i<elements;__i++){
                temp[__i]=elements[__i]*factor;
            }
            SparseMatrix<T> res(rows,cols,temp,rtemp,ctemp);
            free(rtemp);
            free(ctemp);
            free(temp);
            return res;
        }

        // Transpose (returns transposed matrix)
        SparseMatrix<T> transposed() const{
            SparseMatrix<T> trans(cols,rows,elements,cpos,rpos);
            return trans;
        }

        // Sparse Matrix - Vector multiplication : O(N)
        NumVector<T> operator*(const NumVector<T>& that) const{
            assert(that.getSize()==cols);
            T* __temp = (T*)malloc(rows*sizeof(T));
            for(uint32_t __i=0;__i<rows;__i++){
                __temp[__i]=0;
            }
            for(uint32_t __i=0;__i<elements;__i++){
                __temp[rpos[__i]]+=elements[__i]*that[cpos[__i]];
            }
            NumVector<T> __ret(rows,__temp);
            free(__temp);
            return __ret;
        }
        // Sparse Matrix - Vector multiplication : O(N)
        NumVector<T> operator*(NumVector<T>& that) const{
            assert(that.getSize()==cols);
            T* __temp = (T*)malloc(rows*sizeof(T));
            for(uint32_t __i=0;__i<rows;__i++){
                __temp[__i]=0;
            }
            for(uint32_t __i=0;__i<elements;__i++){
                __temp[rpos[__i]]+=elements[__i]*that[cpos[__i]];
            }
            NumVector<T> __ret(rows,__temp);
            free(__temp);
            return __ret;
        }

};

}
#endif
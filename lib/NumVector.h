// #include <Python.h>
// #include "matplotlibcpp.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
using namespace std;

// namespace plt = matplotlibcpp;

namespace numMethods{

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


// Vector class for any number type, assumed to be column vector
template <typename T> class NumVector {
    protected:
        size_t size = 0;
        T *nums;
    public:
        // Constructors
        NumVector() {
        }
        // Empty vector of size sz
        NumVector(uint32_t sz){
            size = sz;
            nums = (T*)malloc(sz*sizeof(T));
            memset(nums, 0, sz*sizeof(T));
        }
        // Vector with size sz
        NumVector(uint32_t sz, T *dataArr){
            size = sz;
            nums = (T*)malloc(sz*sizeof(T));
            memcpy(nums, dataArr, sz*sizeof(T));
        }
        ~NumVector(){
            // free(nums);
        }
        // Get functions
        size_t getSize() const {
            return size;
        }
        T *getNums() const {
            return nums;
        }
        // Access
        // 0-based array access
        T operator[] (int i) const {
            assert(i >= 0 && i < size);
            return *(nums + i);
        }
        // 0-based array access
        T &operator[] (int i) {
            assert(i >= 0 && i < size);
            return *(nums + i);
        }
        // 1-based vector access
        T ele(int i) const{
            assert(i >= 1 && i <= size);
            return *(nums + i-1);
        }
        // 1-based vector access
        T &ele(int i) {
            assert(i >= 1 && i <= size);
            return *(nums + i-1);
        }
        // Assignment
        NumVector<T>& operator= (const NumVector<T>& that) {
            size = that.getSize();
            nums = (T*)malloc(size*sizeof(T));
            memcpy(nums,that.getNums(),size*sizeof(T));
            // nums = that.getNums();
            return *this;
        }
        // Equality check
        bool operator== (NumVector<T> &that) const {
            if(size!=that.getSize()) return false;
            else{
                T *ittr = nums, *ittr2 = that.getNums();
                for(;ittr!=nums+size;ittr++,ittr2++){
                    if((*ittr)!=(*ittr2))return false;
                }
                return true;
            }
        }
        // Vector addition
        NumVector<T> operator+ (NumVector<T> &that) const {
            assert(size == that.getSize());
            T* arr = (T*)malloc(size*sizeof(T));
            T *ittr = nums, *ittr2 = that.getNums(), *ittr3 = arr;
            for (; ittr != nums + size; ittr++,ittr2++,ittr3++){
                (*ittr3) = (*ittr) + (*ittr2);
            }
            NumVector<T> res(size, arr);
            free(arr);
            return res;
        }
        NumVector<T> operator+ (const NumVector<T> &that) const {
            assert(size == that.getSize());
            T* arr = (T*)malloc(size*sizeof(T));
            T *ittr = nums, *ittr2 = that.getNums(), *ittr3 = arr;
            for (; ittr != nums + size; ittr++,ittr2++,ittr3++){
                (*ittr3) = (*ittr) + (*ittr2);
            }
            NumVector<T> res(size, arr);
            free(arr);
            return res;
        }
        // Vector subtraction
        NumVector<T> operator- (NumVector<T> &that) const {
            return (*this) + ((T)(-1) * that);
        }
        NumVector<T> operator- (const NumVector<T> &that) const {
            return (*this) + ((T)(-1) * that);
        }
        // Scaling
        NumVector<T> operator* (T& factor) const {
            T *ittr = nums;
            T* arr = (T*)malloc(size*sizeof(T));
            T* ittr2 = arr;
            for(;ittr != nums+size;ittr++,ittr2++){
                (*ittr2) = factor * (*ittr);
            }
            NumVector<T> res(size, arr);
            free(arr);
            return res;
        }
        NumVector<T> operator* (T const& factor) const {
            T *ittr = nums;
            T* arr = (T*)malloc(size*sizeof(T));
            T* ittr2 = arr;
            for(;ittr != nums+size;ittr++,ittr2++){
                (*ittr2) = factor * (*ittr);
            }
            NumVector<T> res(size, arr);
            free(arr);
            return res;
        }

        // More overloaded assignments...
        NumVector<T>& operator+= (const NumVector<T>& that) {
            *this = *this + that;
            return *this;
        }
        NumVector<T>& operator-= (const NumVector<T>& that) {
            *this = *this - that;
            return *this;
        }
        NumVector<T>& operator*= (T const& factor) {
            *this = *this * factor;
            return *this;
        }
        // Dot product
        T dot(NumVector<T> & that) const {
            assert(size == that.getSize());
            T res = 0;
            T *ittr = nums, *ittr2 = that.getNums();
            for (; ittr != nums + size; ittr++, ittr2++){
                res += (*ittr) * (*ittr2);
            }
            return res;
        }
        // Euclidean norm
        float norm() const {
            T temp = 0;
            T *ittr = nums;
            for(; ittr != nums+size; ittr++){
                temp +=
                (*ittr) * (*ittr);
            }
            return sqrt(temp);
        }
        // Normalisation
        // Normalising the vector, returns a NumVector of floats
        NumVector<float> normalise() const {
            float norm = this->norm();
            assert(norm > 0);
            float *arr = (float*)malloc(size*sizeof(float));
            T *ittr = nums;
            float *ittr2 = arr;
            for(;ittr != nums+size; ittr++,ittr2++){
                (*ittr2) = (float)(*ittr) / norm;
            }
            NumVector<float> res(size,arr);
            free(arr);
            return res;
        }
        // __entry: {index (0 based), value} of first non-zero entry, returns {size, 0} if there are no non-zero entries
        struct __entry {
            uint32_t index;
            T value;
        };
        __entry leading() const {
            T *ittr = nums;
            for(uint32_t pos = 0;ittr!=nums+size;ittr++,pos++){
                if(*ittr!=0)return{pos,(*ittr)};
            }
            return {(uint32_t)size,0};
        }
        // Adjust for floating point errors
        void adjust() {
            for(uint32_t pos=0;pos<size;pos++){
                if(*(nums+pos) == 0.000){
                    *(nums+pos) = +0.0;
                }
            }
        }

        // Standard vectors
        // Zero vector
        template <typename T2> friend NumVector<T2> ZeroVect(size_t sz);
        // en base vector
        template <typename T2> friend NumVector<T2> UnitVect(size_t sz, uint32_t n);

        /*
        TODO: implement fixed size storage of nums (done)
        implement vector arithmetic (addition, dot product, scaling) (done)
        implement euclidean norm (done)
        implement standard vectors (done)
        implement nomralisation of vectors (done)
        */
};
// for commutativity
template <typename T> NumVector<T> operator* (T const& factor, NumVector<T> that) {
    return that * factor;
}
// Zero vector of size sz
template<typename T> NumVector<T> ZeroVect(size_t sz) {
    NumVector<T> res(sz);
    return sz;
}
// en base vector
template <typename T> NumVector<T> UnitVect(size_t sz, uint32_t n){
    NumVector<T> res = ZeroVect<T>(sz);
    res[n] = 1;
    return res;
}
template <typename T> T __firstArg(T a, T b){
    return a;
}


}
       
/*

 */
#include <Python.h>
#include "matplotlibcpp.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
using namespace std;

namespace plt = matplotlibcpp;

namespace numMethods{
// constants
#define EPS numeric_limits<double>::epsilon()

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
            float norm = (*this).norm();
            assert(norm > 0);
            float *arr = (float*)malloc(size*sizeof(float));
            T *ittr = nums;
            float *ittr2 = arr;
            for(;ittr != nums+size; ittr++,ittr2++){
                (*ittr2) = (float)(*ittr) / norm;
            }
            NumVector<float> res(size,arr);
            for(int i=0;i<size;i++){
                std::cerr<<res[i]<<" ";
            }
            std::cerr<<"\n";
            free(arr);
            return res;
        }
        /*
        TODO: implement fixed size storage of nums (done)
        implement vector arithmetic (addition, dot product, scaling) (done)
        implement euclidean norm (done)
        implement standard vectors
        implement nomralisation of vectors (done)
        */
};
// for commutativity
template <typename T> NumVector<T> operator* (T const& factor, NumVector<T> that) {
    return that * factor;
}

// Matrix class for any number type
template <typename T> class Matrix {
    protected:
        size_t rows = 0, cols = 0, elements = 0;
        T *nums[2]; // this implementation is gonna be cursed :fire:
        // Constructor for setting everything directly
        Matrix <T> (uint32_t r, uint32_t c, T *arr1, T *arr2){
            rows = r;
            cols = c;
            elements = r * c;
            nums[0] = (T*)malloc(elements*sizeof(T));
            nums[1] = (T*)malloc(elements*sizeof(T));
            memcpy(nums[0],arr1,elements*sizeof(T));
            memcpy(nums[1],arr2,elements*sizeof(T));
            rind = (uint32_t*)malloc(rows*sizeof(uint32_t));
            cind = (uint32_t*)malloc(cols*sizeof(uint32_t));
            for(int i=0;i<rows;i++){ rind[i]=i; }
            for(int i=0;i<cols;i++){ cind[i]=i; }
        }
        
    public:
        uint32_t *rind, *cind;
        // Constructors
        Matrix(){
        }
        // Empty matrix of size r * c
        Matrix(uint32_t r, uint32_t c){
            rows = r;
            cols = c;
            elements = r * c;
            nums[0] = (T*)malloc(elements*sizeof(T));
            nums[1] = (T*)malloc(elements*sizeof(T));
            memset(nums[0],0,elements*sizeof(T));
            memset(nums[1],0,elements*sizeof(T));
            rind = (uint32_t*)malloc(rows*sizeof(uint32_t));
            cind = (uint32_t*)malloc(cols*sizeof(uint32_t));
            for(int i=0;i<rows;i++){ rind[i]=i; }
            for(int i=0;i<cols;i++){ cind[i]=i; }
        }
        // Matrix of size r * c
        Matrix(uint32_t r, uint32_t c, T* dataArr){
            rows = r;
            cols = c;
            elements = r * c;
            nums[0] = (T*)malloc(elements*sizeof(T));
            nums[1] = (T*)malloc(elements*sizeof(T));
            for(int i=0;i<r;i++){
                for(int j=0;j<c;j++){
                    nums[0][i*c+j]=*(dataArr+i*c+j);
                    nums[1][j*r+i]=*(dataArr+i*c+j);
                }
            }
            rind = (uint32_t*)malloc(rows*sizeof(uint32_t));
            cind = (uint32_t*)malloc(cols*sizeof(uint32_t));
            for(int i=0;i<rows;i++){ rind[i]=i; }
            for(int i=0;i<cols;i++){ cind[i]=i; }
        }
        // Destructor
        ~Matrix(){
            // TO-DO: find a way to deallocate memory
            // free(nums[0]);
            // free(nums[1]);
            // free(nums);
        }
        // Get functions
        size_t getRows() const{
            return rows;
        }
        size_t getCols() const{
            return cols;
        }
        size_t getSize() const{
            return elements;
        }
        T* getNums(int i) const{
            return nums[i];
        }
        // Access
        // 0-based array access (read only): M[r][c]
        class unordered {
            protected:
                T* ind;
                const Matrix<T>& parent;
            public:
                unordered(T* x,const  Matrix<T>& par) : parent(par){
                    ind = x;
                    // parent = par;
                }
                T operator[] (int c) const {
                    assert(c>=0&&c<parent.getCols());
                    return *(parent.cind[c]+ind);
                }
        };
        unordered operator[] (int r) const {
            assert(r>=0&&r<rows);
            unordered temp(nums[0]+rind[r]*cols,*this);
            return temp;
        }
        // Gather row as array
        T* row(int r) const{
            assert(r>=0&&r<rows);
            return nums[0]+r*cols;
        }
        // Gather column as array
        T* col(int c) const {
            assert(c>=0&&c<cols);
            return nums[1]+c*rows;
        }
        // Set particular element
        void set(int r, int c, int val){
            assert(r>=0&&r<rows&&c>=0&&c<cols);
            nums[0][r*cols+c]=val;
            nums[1][c*rows+r]=val;
        }
        // Returns the transpose
        Matrix<T> transposed() const{
            Matrix<T> trans(cols,rows,nums[1],nums[0]);
            return trans;
        }
        // Transpose (in place)
        void transpose() {
            swap(rows,cols);
            swap(nums[0],nums[1]);
        }
        // Returns matrix after row swap
        Matrix<T> rswapped(int i, int j) const{
            assert(i>=0&&i<rows);
            assert(j>=0&&j<rows);
            T* temp[2];
            temp[0] = (T*)malloc(elements*sizeof(T));
            temp[1] = (T*)malloc(elements*sizeof(T));
            memcpy(temp[0],nums[0],elements*sizeof(T));
            memcpy(temp[1],nums[1],elements*sizeof(T));
            for(int k = 0; k < cols;k++){
                swap(temp[0][i*cols+k],temp[0][j*cols+k]);
                swap(temp[1][k*rows+i],temp[1][k*rows+j]);
            }
            Matrix<T> res(rows,cols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        // Row swap (in place)
        void rswap(int i, int j){
            assert(i>=0&&i<rows);
            assert(j>=0&&j<rows);
            for(int k = 0; k < cols;k++){
                swap(nums[0][i*cols+k],nums[0][j*cols+k]);
                swap(nums[1][k*rows+i],nums[1][k*rows+j]);
            }
        }
        // Column swap (in place)
        void cswap(int i, int j){
            assert(i>=0&&i<cols);
            assert(j>=0&&j<cols);
            for(int k = 0; k < rows;k++){
                swap(nums[0][k*cols+i],nums[0][k*cols+j]);
                swap(nums[1][i*rows+k],nums[1][j*rows+k]);
            }
        }
        // Returns matrix after column swap
        Matrix<T> cswapped(int i, int j) const{
            assert(i>=0&&i<cols);
            assert(j>=0&&j<cols);
            T* temp[2];
            temp[0] = (T*)malloc(elements*sizeof(T));
            temp[1] = (T*)malloc(elements*sizeof(T));
            memcpy(temp[0],nums[0],elements*sizeof(T));
            memcpy(temp[1],nums[1],elements*sizeof(T));
            for(int k = 0; k < rows;k++){
                swap(temp[0][k*cols+i],temp[0][k*cols+j]);
                swap(temp[1][i*rows+k],temp[1][j*rows+k]);
            }
            Matrix<T> res(rows,cols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        // Assignment
        Matrix<T>& operator=(const Matrix<T>& that) {
            rows = that.getRows();
            cols = that.getCols();
            elements = rows*cols;
            free(nums[0]);
            free(nums[1]);
            nums[0] = (T*)malloc(elements*sizeof(T));
            nums[1] = (T*)malloc(elements*sizeof(T));
            memcpy(nums[0],that.getNums(0),elements*sizeof(T));
            memcpy(nums[1],that.getNums(1),elements*sizeof(T));
            return (*this);
        }
        // Equality check
        bool operator==(const Matrix<T>& that) const{
            if(rows!=that.getRows()||cols!=that.getCols())return false;
            else{
                for(int i=0;i<elements;i++){
                    if(nums[0][i]!=that.getNums(0)[i])return false;
                }
                return true;
            }
        }
        // Matrix addition
        Matrix<T> operator+(Matrix<T>& that) const{
            assert(rows==that.getRows()&&cols==that.getCols());
            T* temp[2];
            temp[0] = (T*)malloc(elements*sizeof(T));
            temp[1] = (T*)malloc(elements*sizeof(T));
            for(int i=0;i<elements;i++){
                temp[0][i]=nums[0][i]+that.getNums(0)[i];
                temp[1][i]=nums[1][i]+that.getNums(1)[i];
            }
            Matrix<T> res(rows,cols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        Matrix<T> operator+(const Matrix<T>& that) const{
            assert(rows==that.getRows()&&cols==that.getCols());
            T* temp[2];
            temp[0] = (T*)malloc(elements*sizeof(T));
            temp[1] = (T*)malloc(elements*sizeof(T));
            for(int i=0;i<elements;i++){
                temp[0][i]=nums[0][i]+that.getNums(0)[i];
                temp[1][i]=nums[1][i]+that.getNums(1)[i];
            }
            Matrix<T> res(rows,cols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        // Matrix scaling
        Matrix<T> operator*(T const& factor) const{
            T* temp[2];
            temp[0] = (T*)malloc(elements*sizeof(T));
            temp[1] = (T*)malloc(elements*sizeof(T));
            for(int i=0;i<elements;i++){
                temp[0][i]=nums[0][i]*factor;
                temp[1][i]=nums[1][i]*factor;
            }
            Matrix<T> res(rows,cols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        Matrix<T> operator*(T& factor) const{
            T* temp[2];
            temp[0] = (T*)malloc(elements*sizeof(T));
            temp[1] = (T*)malloc(elements*sizeof(T));
            for(int i=0;i<elements;i++){
                temp[0][i]=nums[0][i]*factor;
                temp[1][i]=nums[1][i]*factor;
            }
            Matrix<T> res(rows,cols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        // Matrix subtraction
        Matrix<T> operator-(Matrix<T>& that) const{
            return (*this) + ((T)(-1) * that);
        }
        Matrix<T> operator-(const Matrix<T>& that) const{
            return (*this) + ((T)(-1) * that);
        }
        // Matrix multiplication
        Matrix<T> operator*(Matrix<T>& that) const{
            // non-commutivity is gonna come back to bite me
            assert(cols==that.getRows());
            uint32_t resElems = rows * that.getCols(), resRows=rows, resCols=that.getCols();
            T* temp[2];
            temp[0] = (T*)malloc(resElems*sizeof(T));
            temp[1] = (T*)malloc(resElems*sizeof(T));
            for(int i=0;i<resRows;i++){
                for(int j=0;j<resCols;j++){
                    NumVector<T>r(cols,(*this).row(i)),c(cols,that.col(j));
                    // cerr<<"bruh";
                    T dotRes = r.dot(c);
                    temp[0][i*resCols+j]=dotRes;
                    temp[1][j*resRows+i]=dotRes;
                }
            }
            Matrix<T> res(resRows,resCols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        Matrix<T> operator*(const Matrix<T>& that) const{
            // non-commutivity is gonna come back to bite me
            assert(cols==that.getRows());
            uint32_t resElems = rows * that.getCols(), resRows=rows, resCols=that.getCols();
            T* temp[2];
            temp[0] = (T*)malloc(resElems*sizeof(T));
            temp[1] = (T*)malloc(resElems*sizeof(T));
            for(int i=0;i<resRows;i++){
                for(int j=0;j<resCols;j++){
                    NumVector<T>r(cols,(*this).row(i)),c(cols,that.col(j));
                    // cerr<<"bruh";
                    T dotRes = r.dot(c);
                    temp[0][i*resCols+j]=dotRes;
                    temp[1][j*resRows+i]=dotRes;
                }
            }
            Matrix<T> res(resRows,resCols,temp[0],temp[1]);
            free(temp[0]);
            free(temp[1]);
            return res;
        }
        // More operator overloads
        Matrix<T> operator+=(const Matrix<T>& that){
            (*this) = (*this) + that;
            return (*this);
        }
        Matrix<T> operator-=(const Matrix<T>& that){
            (*this) = (*this) - that;
            return (*this);
        }
        Matrix<T> operator*=(const Matrix<T>& that){
            (*this) = (*this) * that;
            return (*this);
        }
        Matrix<T> operator*=(T const& that){
            (*this) = (*this) * that;
            return (*this);
        }
        // Echelon form reduction (without pivoting)
        Matrix<T> echRedNoPivot() const{
            
        }
        // Pivoting
        Matrix<T> pivot() const{

        }
        // Echelon form reduction (with pivoting)
        Matrix<T> echRed() const{

        }
        // Matrix inverse
        
        /*
        TODO: implement matrix storage (sequence of vectors? 2D array?) (done)
        implement transpose (done)
        implement swapping (done)
        implement matrix arithmetic (addition (done), multiplication (done!!!!), scaling (done))
        implement matrix inverse, determinant
        implement standard matrices
        implement rank
        implement eigenvectors / eigenvalues
        implement diagonalisation
        implement classification
        implement factorisation
        */
};
// (for commutativity) Matrix scaling
template<typename T> Matrix<T> operator*(T const& factor, Matrix<T> that){
    return that * factor;
}
// Column vector class for any number type, supports vector operations (implemented as n * 1 matrix with additional vector arithmetic)
template <typename T> class MatVector : protected Matrix<T> {
    protected:

    public:
        MatVector() : Matrix<T>(){
        }
        MatVector(uint32_t size, T * dataArr) : Matrix<T>(){

        }
};

class Differentiable {
    protected:
    public:
        virtual void differentiate(){
            
        }
};



class Estimator {
    
};

}
       
/*

 */
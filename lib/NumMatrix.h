#include <cassert>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>

#ifndef __NUMMATRIX_H__
#define __NUMMATRIX_H__
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


// Matrix class for any number type
template <typename T> class Matrix {
    protected:
        size_t rows = 0, cols = 0, elements = 0;
        T *nums[2]; // this implementation is gonna be cursed :fire:
        
        public:
        // Constructors
        Matrix(){
        }
        // Constructor for setting everything directly
        Matrix <T> (uint32_t r, uint32_t c, T *arr1, T *arr2){
            rows = r;
            cols = c;
            elements = r * c;
            nums[0] = (T*)malloc(elements*sizeof(T));
            nums[1] = (T*)malloc(elements*sizeof(T));
            memcpy(nums[0],arr1,elements*sizeof(T));
            memcpy(nums[1],arr2,elements*sizeof(T));
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
        T* operator[] (int r) const {
            assert(r>=0&&r<rows);
            return nums[0]+r*cols;
        }
        // Gather row as array
        T* row(int r) const{
            assert(r>=0&&r<rows);
            return nums[0]+r*cols;
        }
        // Extract row as NumVector
        NumVector<T> extractNumRow(int r) const{
            assert(r>=0&&r<rows);
            T* arr = (T*)malloc(cols*sizeof(T));
            memcpy(arr,nums[0]+r*cols,cols*sizeof(T));
            NumVector<T> res(cols,arr);
            return res;
        }
        // Gather column as array
        T* col(int c) const {
            assert(c>=0&&c<cols);
            return nums[1]+c*rows;
        }
        // Extract column as NumVector
        NumVector<T> extractNumCol(int c) const{
            assert(c>=0&&c<cols);
            T* arr = (T*)malloc(rows*sizeof(T));
            memcpy(arr,nums[1]+c*rows,rows*sizeof(T));
            NumVector<T> res(rows,arr);
            return res;
        }
        // Set particular element
        void set(int r, int c, T val){
            assert(r>=0&&r<rows&&c>=0&&c<cols);
            nums[0][r*cols+c]=val;
            nums[1][c*rows+r]=val;
        }
        // Set row r based on array
        void setRow(uint32_t r, T* that){
            assert(r>=0&&r<rows);
            memcpy(nums[0]+r*cols,that,cols*sizeof(T));
            for(uint32_t i=0;i<cols;i++){
                nums[1][i*rows+r]=that[i];
            }
        }
        // Set row r based on NumVector
        void setRow(uint32_t r, NumVector<T> that){
            assert(r>=0&&r<rows);
            assert(that.getSize()==cols);
            memcpy(nums[0]+r*cols,that.getNums(),cols*sizeof(T));
            for(uint32_t i=0;i<cols;i++){
                nums[1][i*rows+r]=that[i];
            }
        }
        // Set column c based on array
        void setCol(uint32_t c, T* that){
            assert(c>=0&&c<cols);
            memcpy(nums[1]+c*rows,that,rows*sizeof(T));
            for(uint32_t i=0;i<rows;i++){
                nums[1][i*cols+c]=that[i];
            }
        }
        // Set column c based on NumVector
        void setCol(uint32_t c, NumVector<T> that){
            assert(c>=0&&c<cols);
            assert(that.getSize()==rows);
            memcpy(nums[1]+c*rows,that,rows*sizeof(T));
            for(uint32_t i=0;i<rows;i++){
                nums[1][i*cols+c]=that[i];
            }
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
                    NumVector<T>r(cols,this->row(i)),c(cols,that.col(j));
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
                    NumVector<T>r(cols,this->row(i)),c(cols,that.col(j));
                    r.adjust();
                    c.adjust();
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
        // Convert to float Matrix
        Matrix<float> convert() const{
            float* arr[2];
            arr[0]=(float*)malloc(elements*sizeof(float));
            arr[1]=(float*)malloc(elements*sizeof(float));
            uint32_t i=0;
            for(i=0;i<elements;i++){
                arr[0][i]=(float)nums[0][i];
                arr[1][i]=(float)nums[1][i];
            }
            Matrix<float>res(rows,cols,arr[0],arr[1]);
            return res;
        }

        // Standard matrices
        // Zero matrix
        template <typename T2> friend Matrix<T2> ZeroMat(size_t dim);
        // Identity matrix
        template <typename T2> friend Matrix<T2> Iden(size_t dim);

        // (when will you even use this)
        // Row echelon form reduction (without pivoting)
        Matrix<T> echRedNoPivot(std::function<T(T,T)> coTarg=__firstArg<T>, std::function<T(T,T)>coLead=__firstArg<T>) const{
            uint32_t i = 0, j = 0, currLead=0;
            Matrix<T> res(rows,cols,nums[0],nums[1]);
            T leadVal=0;
            for(i=0;i<rows;i++){
                // Extract current row
                NumVector<T>curr = res.extractNumRow(i);
                // Check that it leads
                assert(curr.leading().index>=currLead);
                currLead = curr.leading().index;
                leadVal = curr.leading().value;
                if(leadVal){
                    // Go down and eliminate every row below
                    for(j=i+1;j<rows;j++){
                        NumVector<T>target = res.extractNumRow(j);
                        T targLead=target[currLead];
                        T uValTarg = coTarg(leadVal,targLead), uValCurr = coLead(targLead, leadVal); // uValTarg: update coefficient for target row, uValCurr: update coefficient for leading row
                        // (this is bad don't use this)
                        target = uValTarg * target - uValCurr * curr;
                        res.setRow(j,target);
                        target.adjust();
                    }
                }
            }
            return res;
        }

        // (just use this)
        // Echelon form reduction (with pivoting)
        Matrix<T> echRed(std::function<T(T,T)> coTarg=__firstArg<T>, std::function<T(T,T)>coLead=__firstArg<T>) const{
            uint32_t i = 0, j = 0, currLead=0, k=0;
            Matrix<T> res(rows,cols,nums[0],nums[1]);
            T leadVal=0;
            k = 0;
            for(i=0;i<rows;i++){
                for(j=k;j<rows;j++){
                    if(res[j][i])break;
                }
                if(j==rows)continue;
                if(j!=k)res.rswap(k,j);
                NumVector<T>curr = res.extractNumRow(k);
                assert(curr.leading().index>=currLead);
                currLead = curr.leading().index;
                leadVal = curr.leading().value;
                if(leadVal){
                    for(j=k+1;j<rows;j++){
                        NumVector<T>target = res.extractNumRow(j);
                        T targLead=target[currLead];
                        T uValTarg = coTarg(leadVal,targLead), uValCurr = coLead(targLead, leadVal); // uValTarg: update coefficient for target row, uValCurr: update coefficient for leading row
                        target = uValTarg * target - uValCurr * curr;
                        res.setRow(j,target);
                        target.adjust();
                    }
                }
                k++;
            }
            return res;
        }

        // Reduced row Echelon form reduction (with pivoting)
        Matrix<float> RREF() const{
            int i = 0, j = 0, currLead=0, k=0;
            Matrix<float> res = this->convert();
            float leadVal=0;
            k = 0;
            for(i=0;i<rows;i++){
                for(j=k;j<rows;j++){
                    if(res[j][i])break;
                }
                if(j==rows)continue;
                if(j!=k)res.rswap(k,j);
                NumVector<float>curr = res.extractNumRow(k);
                assert(curr.leading().index>=currLead);
                currLead = curr.leading().index;
                leadVal = curr.leading().value;
                if(leadVal){
                    for(j=k+1;j<rows;j++){
                        NumVector<float>target = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        float uValCurr = targLead/leadVal;
                        target -= uValCurr * curr;
                        target.adjust();
                        res.setRow(j,target);
                    }
                }
                k++;
            }
            // Reducing leading variables and converting to RREF
            for(i=0;i<rows;i++){
                NumVector<float>curr = res.extractNumRow(i);
                leadVal = curr.leading().value;
                currLead = curr.leading().index;
                if(leadVal!=0){
                    curr *= (1.0/(float)curr.leading().value);
                    curr.adjust();
                    res.setRow(i,curr);
                    for(j=i-1;j>=0;j--){
                        NumVector<float>target = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        target -= targLead * curr;
                        target.adjust();
                        res.setRow(j,target);
                    }
                }
            }
            return res;
        }

        // (Gaussian reduction is OP)
        // Determinant
        float determinant() const{
            if(rows!=cols) return 0;
            int i = 0, j = 0, currLead=0, k=0;
            float det = 1;
            Matrix<float> res = this->convert();
            float leadVal=0;
            k = 0;
            for(i=0;i<rows;i++){
                for(j=k;j<rows;j++){
                    if(res[j][i])break;
                }
                if(j==rows)continue;
                if(j!=k){
                    det *= -1;
                    res.rswap(k,j);
                }
                NumVector<float>curr = res.extractNumRow(k);
                assert(curr.leading().index>=currLead);
                currLead = curr.leading().index;
                leadVal = curr.leading().value;
                if(leadVal){
                    for(j=k+1;j<rows;j++){
                        NumVector<float>target = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        float uValCurr = targLead/leadVal;
                        target -= uValCurr * curr;
                        target.adjust();
                        res.setRow(j,target);
                    }
                }
                k++;
            }
            for(i=0;i<rows;i++){
                NumVector<float>curr = res.extractNumRow(i);
                leadVal = curr.leading().value;
                currLead = curr.leading().index;
                if(leadVal!=0){
                    curr *= (1.0/(float)leadVal);
                    det *= (float)leadVal;
                    curr.adjust();
                    res.setRow(i,curr);
                    for(j=i-1;j>=0;j--){
                        NumVector<float>target = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        target -= targLead * curr;
                        target.adjust();
                        res.setRow(j,target);
                    }
                }
            }
            // If there's a zero row -> det = 0
            for(i=0;i<rows;i++){
                det *= res[i][i];
            }
            return det;
        }
        // Matrix inverse
        Matrix<float> inverse() const {
            assert(this->determinant() != 0.0);
            int i = 0, j = 0, currLead=0, k=0;
            Matrix<float> temp = (this->convert()), res = Iden<float>(rows);
            float leadVal=0;
            k = 0;
            for(i=0;i<rows;i++){
                for(j=k;j<rows;j++){
                    if(temp[j][i])break;
                }
                if(j==rows)continue;
                if(j!=k){
                    res.rswap(k,j);
                    temp.rswap(k,j);
                }
                NumVector<float>curr = temp.extractNumRow(k), currInv = res.extractNumRow(k);
                assert(curr.leading().index>=currLead);
                currLead = curr.leading().index;
                leadVal = curr.leading().value;
                if(leadVal){
                    for(j=k+1;j<rows;j++){
                        NumVector<float>target = temp.extractNumRow(j),targInv = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        float uValCurr = targLead/leadVal;
                        target -= uValCurr * curr;
                        targInv -= uValCurr * currInv;
                        target.adjust();
                        targInv.adjust();
                        temp.setRow(j,target);
                        res.setRow(j,targInv);
                    }
                }
                k++;
            }
            for(i=0;i<rows;i++){
                NumVector<float>curr = temp.extractNumRow(i), currInv = res.extractNumRow(i);
                leadVal = curr.leading().value;
                currLead = curr.leading().index;
                if(leadVal!=0){
                    curr *= (1.0/(float)leadVal);
                    curr.adjust();
                    currInv *= (1.0/(float)leadVal);
                    currInv.adjust();
                    temp.setRow(i,curr);
                    res.setRow(i,currInv);
                    for(j=i-1;j>=0;j--){
                        NumVector<float>target = temp.extractNumRow(j),targInv = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        target -= targLead * curr;
                        target.adjust();
                        targInv -= targLead * currInv;
                        targInv.adjust();
                        temp.setRow(j,target);
                        res.setRow(j,targInv);
                    }
                }
            }
            return res;
        }
        // Matrix rank
        uint32_t rank() const {
            int i = 0, j = 0, currLead=0, k=0;
            Matrix<float> res = this->convert();
            float leadVal=0;
            k = 0;
            for(i=0;i<rows;i++){
                for(j=k;j<rows;j++){
                    if(res[j][i])break;
                }
                if(j==rows)continue;
                if(j!=k)res.rswap(k,j);
                NumVector<float>curr = res.extractNumRow(k);
                assert(curr.leading().index>=currLead);
                currLead = curr.leading().index;
                leadVal = curr.leading().value;
                if(leadVal){
                    for(j=k+1;j<rows;j++){
                        NumVector<float>target = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        float uValCurr = targLead/leadVal;
                        target -= uValCurr * curr;
                        target.adjust();
                        res.setRow(j,target);
                    }
                }
                k++;
            }
            for(i=0;i<rows;i++){
                NumVector<float>curr = res.extractNumRow(i);
                leadVal = curr.leading().value;
                currLead = curr.leading().index;
                if(leadVal!=0){
                    curr *= (1.0/(float)curr.leading().value);
                    curr.adjust();
                    res.setRow(i,curr);
                    for(j=i-1;j>=0;j--){
                        NumVector<float>target = res.extractNumRow(j);
                        float targLead=target[currLead];
                        if(targLead==0)continue;
                        target -= targLead * curr;
                        target.adjust();
                        res.setRow(j,target);
                    }
                }
            }
            uint32_t nonZeroRows = 0, leader = 0;
            for(nonZeroRows=0;nonZeroRows<rows;nonZeroRows++){
                NumVector<float>curr = res.extractNumRow(i);
                leadVal = curr.leading().value;
                if(leadVal==0)break;
            }
            return nonZeroRows;
        }

        // Pivot, Lower-triangular, Upper-triangular
        struct PLU{
            Matrix<float> P(), L(), U();
        };
        // PLU Factorisation (pivoting only prevents division by 0)
        PLU PLUfactorize() {
            assert(rows==cols);

        }
        /*
        TODO: implement matrix storage (sequence of vectors? 2D array?) (done)
        implement transpose (done)
        implement swapping (done)
        implement matrix arithmetic (addition (done), multiplication (done!!!!), scaling (done))
        implement row reduction (done)
        implement matrix inverse (done?), determinant (done?)
        implement standard matrices (done)
        implement rank (done)
        implement eigenvectors / eigenvalues
        implement diagonalisation
        implement classification
        implement factorisation (PLU (in progress), QR)
        implement matrix norm
        implement SVD
        */
};
// (for commutativity) Matrix scaling
template<typename T> Matrix<T> operator*(T const& factor, Matrix<T> that){
    return that * factor;
}
// Standard matrices
// Zero matrix
template <typename T> Matrix<T> ZeroMat(size_t dim){
    Matrix<T>res(dim,dim);
    return res;
}
// Identity matrix
template <typename T> Matrix<T> Iden(size_t dim){
    Matrix<T>res(dim,dim);
    for(int i=0;i<dim;i++){
        res.set(i,i,1);
    }
    return res;
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

}
#endif
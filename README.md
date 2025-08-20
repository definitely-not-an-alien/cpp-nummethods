# Numerical Methods in C++

This is a C++ header file for linear algebra, numerical integration, and numerical optimisation / root finding algorithms.

## Documentation

### Contents
- [Linear Algebra](#linear-algebra)
  - [`NumVector` class](#numvector-class)
    - [Constructors](#constructors)
    - [Methods](#methods)
      - [`getSize()`](#getsize)
      - [`getNums()`](#getnums)
      - [`operator[]`](#operator)
      - [`ele()`](#ele)
      - 
  - [`Matrix` class](#matrix-class)
    - [Constructors](#constructors-1)
- [Numerical Calculus](#numerical-calculus)

  * `Differentiable` class
- [Numerical Optimisation](#numerical-optimisation)

## Linear Algebra
### ```NumVector``` class
Class for numerical vectors, assumed to be column vector.
Indices of vector entries are 0-based.

#### Constructors
```c++
NumVector(uint32_t sz)
```
Creates an empty `NumVector` object of size `sz`.

```c++
NumVector(uint32_t sz, T *dataArr)
```
Creates an empty `NumVector` object of size `sz` taking values of type `T` from `dataArr`. Order of values follows that of `dataArr`.

#### Methods

##### getSize()
```c++
size_t getSize()
```
Number of entries in the `NumVector` object - defaults to `0` if entries have not been initialised.

##### getNums()
```c++
T* getNums()
```
A pointer of type `T*` to all entries in the vector / an array of type `T` containing all entries in the vector.

##### operator[]
0-based vector access
```c++
T operator[](int i)
```
Returns the value of the `i`th element (0-based) (0 <= i < size of vector).

```c++
T& operator[](int i)
```
Returns the `i`th element (0-based) by reference.

##### ele()
1-based vector access
```c++
T ele(int i)
```
Returns the value of the `i`th element (1-based) (1 <= i <= size of vector).
```c++
T& ele(int i)
```
Returns the `i`th element (1-based) by reference.



### ```Matrix``` class
Class for numerical matrices

#### Constructors


## Numerical Calculus

## Numerical Optimisation

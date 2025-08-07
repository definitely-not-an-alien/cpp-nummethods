# Numerical Methods in C++

This is a C++ header file for linear algebra, numerical integration, and numerical optimisation / root finding algorithms.

## Documentation

### Contents
- [Linear Algebra](#linear-algebra)
- - `NumVector` class
- - - [Constructors](#constructors)
- - - [Methods](#methods)
- - - - [`getSize()`](#getsize)
- - `Matrix` class
- - - [Constructors](#constructors-1)
- [Numerical Calculus](#numerical-calculus)

- * `Differentiable` class
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
getSize()
```
Number of entries in the `NumVector` object - defaults to 0 if entries have not been initialised.


### ```Matrix``` class
Class for numerical matrices

#### Constructors


## Numerical Calculus

## Numerical Optimisation

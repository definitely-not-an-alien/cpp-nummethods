# Numerical Methods in C++

This is a C++ header file for linear algebra, numerical integration, and numerical optimisation / root finding algorithms.

## Documentation

### Contents
- [Using the package](#using-the-package)
- [`numMethods` namespace](#nummethods-namespace)
- [Linear Algebra](#linear-algebra)
  - [`NumVector` class](#numvector-class)
    - [Constructors](#constructors)
    - [Methods](#methods)
      - [`getSize()`](#getsize)
      - [`getNums()`](#getnums)
      - [`operator[]`](#operator)
      - [`ele()`](#ele)
      - [`operator=`](#operator-1)
      - [`operator==`](#operator-2)
      - [`operator+`](#operator-3)
      - [`operator-`](#operator-)
      - [`operator*`](#operator-4)
      - [`operator+=`, `operator-=`, `operator*=`](#operator-operator--operator)
      - [`dot()`](#dot)
      - [`norm()`](#norm)
      - [`normalise()`](#normalise)
      - [`leading()`](#leading)
      - [`adjust()`](#adjust)
    - [Standard Vectors]
      - [`ZeroVect`]
      - [`UnitVect`]
  - [`Matrix` class](#matrix-class)
    - [Constructors](#constructors-1)
- [Numerical Calculus](#numerical-calculus)

  * `Differentiable` class
- [Numerical Optimisation](#numerical-optimisation)

## Using the Package 
<details open>
To use the package, download the `lib` folder containing all of the header files and place it into your project directory (or another folder added into your include path).

All relevant header files are included in the header `NumMethodsCpp.h`, so adding the line 
```c++
#include "NumMethodsCpp.h"
```
will be sufficient.

The `numMethods` namespace will have to be accessed to use the implemented functions.
</details>

## ```numMethods``` namespace

<details>
Everything in this package is wrapped inside the `numMethods` namespace.

To access a particular class or function, the `numMethods` namespace must be accessed.

Example:
```c++
#include "NumMethodsCpp.h"
numMethods::NumVector<int> a(5); // Empty integer NumVector object of size 5.
```
whereas
```c++
NumVector<int> a(5);
```
might not compile.

For convenience's sake, you can write
```c++
using namespace numMethods;
```
but as with using other namespaces, there may be risks of name clashings.

</details>

## Linear Algebra
### ```NumVector``` class
Class for numerical vectors, assumed to be column vector.
Indices of vector entries are 0-based.
<details open>

#### Constructors
```c++
NumVector(uint32_t sz)
```
Creates an empty `NumVector` object of size `sz`.

```c++
NumVector(uint32_t sz, T *dataArr)
```
Creates an empty `NumVector` object of size `sz` taking values of type `T` from `dataArr`. Order of values follows that of `dataArr`.

Usage:
```c++
int arr[5] = {0,1,2,3,4};
numMethods::NumVector<int> a(5,arr);
```

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

##### operator=
Assignment
```c++
NumVector<T>& operator= (const NumVector<T>& that)
```
Assigns the values and size of a `NumVector` object to take those of the argument `that`.

Usage:
```c++
int arr[5] = {0,1,2,3,4};
numMethods::NumVector<int> a(5,arr);
numMethods::NumVector<int> b = a; // b has size 5 and contains {0,1,2,3,4}
```

##### operator==
```c++
bool operator== (NumVector<T> &that)
```
Equality check for two `NumVector<T>` objects. Returns `true` if the size and values of both objects are equal, returns `false` otherwise (order of elements matter).

_Note: the types `T` of the two objects must be equal_

Usage:
```c++
int arr[5] = {0,1,2,3,4};
int arr2[4] = {0,1,2,3};
int arr3[5] = {4,3,2,1,0};
numMethods::NumVector<int> a(5,arr), b(4,arr2), c(5,arr3), d(5,arr);

bool res1 = (a==b); // false
bool res2 = (a==c); // false
bool res3 = (a==d); // true
```

##### operator+
```c++
NumVector<T> operator+(NumVector<T> &that)
```
```c++
NumVector<T> operator+(const NumVector<T> &that)
```

Performs vector addition on two `NumVector<T>` objects and returns the resulting `NumVector<T>`. Addition is performed left-to-right, entry by entry (i.e. `res[i] = this[i] + that[i]` where `res` is the returned vector of `this + that`).

_Note: the sizes and types of the two vectors must be equal._

##### operator-
```c++
NumVector<T> operator-(NumVector<T> &that)
```
```c++
NumVector<T> operator-(const NumVector<T> &that)
```

Performs vector subtraction on two `NumVector<T>` objects and returns the resulting `NumVector<T>`. Subtraction is performed left-to-right, entry by entry (i.e. `res[i] = this[i] - that[i]` where `res` is the returned vector of `this - that`).

_Note: the sizes and types of the two vectors must be equal._


##### operator*

```c++
NumVector<T> operator*(T& factor)
```
```c++
NumVector<T> operator*(T const& factor)
```

Performs vector scaling on a `NumVector<T>` objects by some `factor` of type `T` and returns the resulting `NumVector<T>`. Scaling is performed entry by entry (i.e. `res[i] = this[i] * factor` where `res` is the returned vector of `this * factor`).

_Note: the types of `this` and `factor` must be equal._

##### operator+=, operator-=, operator*=

Overloaded assignment operators

```c++

```

##### dot()
```c++
T dot(NumVector<T> &that)
```
Returns the dot product of two `NumVector<T>` objects.

##### norm()
```c++
float norm()
```
Returns the Euclidean norm of the `NumVector<T>` object as a `float`.

##### normalise()
```c++
NumVector<float> normalise()
```
Returns the normalised vector of the `NumVector<T>` object as a `NumVector<float>` object.

##### leading()
Return type `__entry`:
```c++
struct __entry {
  uint32_t index;
  T value;
}
```
```c++
__entry leading()
```
Returns the `{index, value}` of the first non-zero entry (index being 0-based).

Returns `{size, 0}` if there are no non-zero entries.

##### adjust()
```c++
void adjust()
```
In-place adjustment of the vector for floating point errors (i.e. `-0`).

</details>

### ```Matrix``` class
Class for numerical matrices

<details>

#### Constructors
</details>

## Numerical Calculus

## Numerical Optimisation

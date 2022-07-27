/****************************************************************
 ****************************************************************
 ****
 **** This text file is part of the source of 
 **** `Introduction to High-Performance Scientific Computing'
 **** by Victor Eijkhout, copyright 2012-2021
 ****
 **** Deep Learning Network code 
 **** copyright 2021 Ilknur Mustafazade
 ****
 ****************************************************************
 ****************************************************************/

#ifndef SRC_VECTOR_H
#define SRC_VECTOR_H
#include <algorithm>
#include <vector>
#include <cassert>
#include <cmath>

class VectorBatch; // forward for friending
class Matrix; // forward for friending
class Vector {
  friend class VectorBatch;
  friend class Matrix;
private:
    std::vector<float> vals;
public:
    Vector();
    Vector( int n )
	: vals(std::vector<float>(n)) {};
    Vector( std::vector<float> vals );
    Vector(int size, int init);
    int size() const;
    void show();
    void add( const Vector &v1);
    void set_ax( float a, Vector &x );
  /*
   * Access
   */
  std::vector<float>& values() { return vals; };
  const std::vector<float>& values() const { return vals; };
  float *data() { return vals.data(); };
  const float *data() const { return vals.data(); };
  float& operator[](int i) { return vals[i]; };
  float operator[](int i) const { return vals[i]; };

    void copy_from( const VectorBatch& );
    void zeros();
    void square();
    Vector operator-(); // Unary negate operator
    Vector& operator=(const Vector& m2); // Copy constructor
    Vector operator+(const Vector &m2); // Element-wise addition
    Vector operator*(const Vector &m2); // Hadamard Product Element-wise multiplication
    Vector operator/(const Vector &m2); // Element-wise division
    Vector operator/=(float x); 
    Vector operator-(const Vector &m2); // Element-wise subtraction
    friend Vector operator-(const float &c, const Vector &m); // for constant
    friend Vector operator*(const float &c, const Vector &m); // for constant-matrix multiplication
    //friend Vector operator/(const Vector &m, const float &c); // for matrix-constant division

  //! Test that all elements are positive
  bool positive() const {
    return all_of
      ( vals.begin(),vals.end(),
	[] (float e) { return e>0; }
	);
  }
};

#endif //SRC_VECTOR_H

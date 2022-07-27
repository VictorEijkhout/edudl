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

#ifndef CODE_VEC2_H
#define CODE_VEC2_H

#include <vector>
#include "vector.h"
#include "matrix.h"
#include <iostream>
#ifdef BLISNN
#include "blis/blis.h"
#endif

#ifdef USE_GSL
#include "gsl/gsl-lite.hpp"
#endif

#define INDEXr(i,j,m,n) (i)*(n)+(j)
#define INDEXc(i,j,m,n) (i)+(j)*(m)

class VectorBatch{
  friend class Matrix;
  friend class Vector;

private: //private:
  std::vector<float> vals;
  int nvectors{0},vector_size{0};
public:
    VectorBatch();
    VectorBatch( int itemsize );
    // this one is in the blis/reference file
    VectorBatch(int nRows, int nCols, bool rand=false);
    VectorBatch( const Vector& );
    void allocate(int,int);

    int size() const { return vals.size(); };
  //! resize the values vector 
    void resize(int m,int n) {
      nvectors = m; set_item_size(n); //r = m; c = n;
      vals.resize(m*n); };
  //! Frobenius norm
    float normf() const {
      float norm{0.f}; int count{0};
      for ( auto e : vals ) {
	norm += e*e; count++;
      }
      //std::cout << "norm squared over " << count << " elements: " << norm << "\n";
      return sqrt(norm);
    };
  //! Test that all elements are positive
    bool positive() const {
      return all_of
	( vals.begin(),vals.end(),
	  [] (float e) { return e>0; }
	);
    }
  //! Test that there are no NaN elements
    bool notnan() const {
      return all_of
	( vals.begin(),vals.end(),
	  [] (float e) { return not isnan(e); }
	);
    }
  //! Test that there are no Inf elements
    bool notinf() const {
      return all_of
	( vals.begin(),vals.end(),
	  [] (float e) { return not isinf(e); }
	);
    }
  //! The size of any vector in the batch
    int item_size() const { return vector_size; };
  //! Set the vector size \todo should be private, maybe eliminated?
    void set_item_size(int n) { vector_size = n; };
  //! How many vectors are there in this batch
    int batch_size() const { return nvectors; };
  //! Set the batch size \todo should be private, maybe eliminated?
    void set_batch_size(int n) { nvectors = n; };
  //! Total number of eleements in the whole batch
    int nelements() const {
      int n = vals.size();
      assert( n==item_size()*batch_size() );
      return n;
    };

  //! Return the values vector, for write
    std::vector<float>& vals_vector() { return vals; };
  //! Return the values vector, for read only
    const std::vector<float>& vals_vector() const { return vals; };
  //! Return the float data. This is for the BLAS interface
    float *data() { return vals.data(); };
  //! Return the float data. This is for the BLAS interface
    const float *data() const { return vals.data(); };
  //! Return the float data with an offset. This is for the BLAS interface
    const float *data( int disp ) const { return vals.data()+disp; };

	
    void v2mp( const Matrix &x, VectorBatch &y) const;
    void v2tmp( const Matrix &x, VectorBatch &y ) const;
    void v2mtp( const Matrix &x, VectorBatch &y ) const;
    void outer2( const VectorBatch &x, Matrix &y ) const;
	
  void add_vector( const std::vector<float> &v );

    /*
     * Indexing
     */
  //! Return element i of the j'th vector in the batch, with bound checking
    float at(int i,int j) const {
	assert( i>=0 ); assert( i<vector_size );
	assert( j>= 0 ); assert( j<nvectors );
	return *data( i + j*vector_size );
    };
    void set_col(int j,const std::vector<float> &v );
    std::vector<float> get_col(int j) const;
    void set_row( int j, const std::vector<float> &v );
    std::vector<float> get_row(int j) const;
    std::vector<float> extract_vector(int v) const;
#ifdef USE_GSL
  gsl::span<float> get_vector(int v);
  //  const gsl::span<float> get_vector(int v) const;
  void set_vector( const gsl::span<float> &v, int j);
#else
  std::vector<float> get_vector(int v) const;
#endif
  void set_vector( const Vector &v, int j);
  void set_vector( const std::vector<float>&v, int j);

  //! Extract one vector from the batch
  Vector get_vectorObj(int j) const {
    Vector vec(vector_size);
    std::copy( vals.begin()+j*vector_size,vals.begin()+(j+1)*vector_size,
	       vec.vals.begin() );
    // for (int i = 0; i < vector_size; i++ ) 
    //   vec.vals.at(i) = vals.at( j * vector_size + i );
    return vec;
  }


  void show() const;
  void display(std::string) const;

  //! Copy a whole batch into this one
  void copy_from( const VectorBatch& in ) {
    assert( vals.size()==in.vals.size() );
    std::copy( in.vals.begin(),in.vals.end(),vals.begin() );
    // for ( int i=0; i<vals.size(); i++)
    //   vals[i] = in.vals[i];
  };

  void addh(const Vector &y);
  void addh(const VectorBatch &y);
  Vector meanh() const;
	
  VectorBatch operator-(); // Unary negate operator
  VectorBatch& operator=(const VectorBatch& m2); // Copy constructor
  void hadamard(const VectorBatch& m1,const VectorBatch& m2);
  VectorBatch operator*(const VectorBatch &m2); // Hadamard Product Element-wise multiplication
  VectorBatch operator/(const VectorBatch &m2); // Element-wise division
  void scaleby( float );
  VectorBatch operator-(const VectorBatch &m2) const; // Element-wise subtraction
  friend VectorBatch operator/(const VectorBatch &m, const float &c); // for matrix-constant division
  friend VectorBatch operator*(const float &c, const VectorBatch &m); // for matrix-constant division
};


#endif

#if 0
  // hm. this doesn't work
  friend void relu_io    (const Vector &i, Vector &v);
  friend void sigmoid_io (const Vector &i, Vector &v);
  friend void softmax_io (const Vector &i, Vector &v);
  friend void none_io    (const Vector &i, Vector &v);
  friend void reluGrad_io(const Vector &m, Vector &a);
  friend void sigGrad_io (const Vector &m, Vector &a);

  friend void relu_io    (const VectorBatch &i, VectorBatch &v);
  friend void sigmoid_io (const VectorBatch &i, VectorBatch &v);
  friend void softmax_io (const VectorBatch &i, VectorBatch &v);
  friend void none_io    (const VectorBatch &i, VectorBatch &v);
  friend void reluGrad_io(const VectorBatch &m, VectorBatch &a);
  friend void sigGrad_io (const VectorBatch &m, VectorBatch &a);
#endif


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

#include <iostream>
using std::cout;
using std::endl;
#include <iomanip>
using std::setprecision;
#include <string>
using std::string;
#include <vector>
#include <algorithm>
#include <cassert>
using std::vector;

#include "trace.h"
#include "vector2.h"

/*
 * These are the vector batch routines that do not have a optimized implementation,
 * such as using BLIS.
 */

VectorBatch::VectorBatch() { // Default constructor
}

//! Construct and set the vector size
VectorBatch::VectorBatch( int vs ) {
  allocate(0,vs);
}

//! Construct from a single vector
VectorBatch::VectorBatch( const Vector& v ) {
    int s = v.size();
    allocate(1,s);
    copy( v.vals.begin(),v.vals.end(),vals.begin() );
    // for ( int i=0; i<s; i++ )
    // 	vals[i] = v[i];
};

//! resize the vals array
void VectorBatch::allocate(int batchsize,int itemsize) {
  // we allow a batchsize of zero
  assert(batchsize>=0);
  assert(itemsize>0);
  vals.resize(batchsize*itemsize);
  set_batch_size(batchsize);
  set_item_size(itemsize);
};

//! Crude output
void VectorBatch::display( string header) const {
  cout << header << "\n";
  for (int j=0; j<batch_size(); j++) {
    for (int i=0; i<item_size(); i++)
      cout << setprecision(5)
	   << vals_vector().at( INDEXc(i,j,item_size(),batch_size()) )
	   << " ";
    cout << "\n";
  }
};

//! Copy a vector into batch j \todo rename to set_vector?
void VectorBatch::set_col(int j,const std::vector<float> &v ) {
  assert( j<batch_size() );
  assert( v.size()==item_size() );
  for (int i = 0; i<nvectors; i++) {
    vals.at( j + vector_size * i ) = v.at(i);
  };
};

//! Extend the batch with a new vector \todo use the allocate method
void VectorBatch::add_vector( const std::vector<float> &v ) {
  const int Nelements = vals.size();
  const int vector_length = v.size();
  if (Nelements==0)
    set_item_size(vector_length);
  else {
    assert( vector_length==item_size() );
    assert( Nelements%vector_length==0 );
  }
  const int m = Nelements/vector_length;
  vals.resize(Nelements+vector_length); nvectors++;
  for (int i = 0; i < vector_length; i++) {
    vals.at( m * vector_length + i ) = v.at(i);
  };
};

std::vector<float> VectorBatch::get_col(int j) const {
  assert( j<batch_size() );
  const int c = item_size();
  std::vector<float> col(c);
  for (int i=0; i<c; i++)
    col.at(i) = vals.at( j + c*i );
  return col;
};

void VectorBatch::set_row( int j, const std::vector<float> &v ) {
  const int c = item_size();
  assert( v.size()==c );
  for (int i = 0; i < c; i++) {
    vals.at( j * c + i ) = v.at(i);
  };
}

std::vector<float> VectorBatch::get_row(int j) const {
  assert( j<batch_size() );
  const int n = item_size();
  std::vector<float> row(n);
  for (int i = 0; i < n; i++)
    row.at(i) = vals.at( j * n + i );
  return row;
};

std::vector<float> VectorBatch::extract_vector(int j) const {
  assert( j<batch_size() );
  const int m = item_size();
  std::vector<float> v(m);
  for (int i = 0; i < m; i++)
    v.at(i) = vals.at( INDEXc(i,j,m,batch_size()) );  // (j * n + i );
  return v;
  //  return get_row(v);
};
#ifdef USE_GSL
gsl::span<float> VectorBatch::get_vector(int v) {
  const int c = item_size();
  return gsl::span<float>( &vals[v*c], c );
};
// const gsl::span<float> VectorBatch::get_vector(int v) const {
//   const int c = item_size();
//   return gsl::span<float>( data(v*c) /* &vals[v*c] */, c );
// };
#else
std::vector<float> VectorBatch::get_vector(int v) const {
  return extract_vector(v);
};
#endif

//! Copy vector elements into vector j of this batch
void VectorBatch::set_vector( const Vector &v, int j) {
  set_vector( v.vals,j );
}

//! Copy vector elements into vector j of this batch
void VectorBatch::set_vector( const vector<float> &v, int j) {
  const int c = item_size();
  assert( v.size()==c );
  for (int i=0; i<c; i++)
    vals.at( j * c + i ) = v.at(i); 
}

void VectorBatch::addh(const Vector &y) { // Add y to every row
  const int r = item_size(), c = batch_size(); 
  assert( r==y.size() );
  for (int j=0; j<c; j++ ) {
    for (int i=0; i<r; i++) {
      vals.at( INDEXc(i,j,r,c) ) += y.vals[i];
    }
  }
}

// void VectorBatch::add(const VectorBatch &y) { // Add y to every row
//   const int r = item_size(), c = batch_size(); 
//   assert( r==y.size() );
//   asserr( c==y.batch_size() );
//   for (int j=0; j<c; j++ ) {
//     for (int i=0; i<r; i++) {
//       vals.at( INDEXc(i,j,r,c) ) += y.vals.at( INDEXc(i,j,r,c) );
//     }
//   }
// }

Vector VectorBatch::meanh() const { // Returns a vector of row-wise means
  const int r = item_size(), c = batch_size();
  Vector mean(r, 0);
  for (int i=0; i<r; i++) {
    float avg = 0.f;
    for ( int j=0; j<c; j++ ) {
      avg += vals.at( INDEXc(i,j,r,v) );
    }
    mean.vals[i] = avg/static_cast<float>( item_size() );
  }
  return mean;
}


/*
 * VLE dangerous. et rid of this one
 */
VectorBatch &VectorBatch::operator=(const VectorBatch &m2) { // Overloading the = operator
  set_item_size( m2.item_size() );
  set_batch_size( m2.batch_size() );

  this->vals = m2.vals; // IM Since we're using vectors we can just use the assignment from that

  return *this;
}


VectorBatch VectorBatch::operator-(const VectorBatch &m2) const {
  assert( item_size()==m2.item_size() );
  assert( batch_size()==m2.batch_size() );
  const int c = m2.item_size(), r = m2.batch_size();
  VectorBatch out(r, c, 0);
    for (int i = 0; i < r * c; i++) {
        out.vals[i] = this->vals[i] - m2.vals[i];
    }
    return out;
}

VectorBatch VectorBatch::operator*(const VectorBatch &m2) { // Hadamard product
  assert( item_size()==m2.item_size() );
  assert( batch_size()==m2.batch_size() );
  const int c = m2.item_size(), r = m2.batch_size();
  VectorBatch out(r, c, 0);
  for (int i = 0; i < nelements(); i++) {
        out.vals[i] = this->vals[i] * m2.vals[i];
    }
    return out;
}

void VectorBatch::hadamard(const VectorBatch& m1,const VectorBatch& m2) {
  const int r = item_size(), c = batch_size();
  assert( r==m1.item_size() ); assert( c==m1.batch_size() );
  assert( r==m2.item_size() ); assert( c==m2.batch_size() );

  const auto& m1vals = m1.vals_vector();
  const auto& m2vals = m2.vals_vector();
  for (int i=0; i<r*c; i++) {
    vals.at(i) = m1vals.at(i) * m2vals.at(i);
    if ( isinf(vals.at(i)) )
      cout << "inf from " << m1vals.at(i) << " * " << m2vals.at(i) << "\n";
  }
  if (trace_progress()) {
    assert( m1.notinf() ); assert( m1.notnan() ); assert( m1.normf()!=0.f );
    assert( m2.notinf() ); assert( m2.notnan() ); assert( m2.normf()!=0.f );
    assert( this->notinf() ); assert( this->notnan() ); assert( this->normf()!=0.f );
  }
  if (trace_scalars()) {
    cout << "delta: "
	 << m1.normf() << "x" << m2.normf() << " => " << this->normf() << "\n";
  }
}

VectorBatch VectorBatch::operator/(const VectorBatch &m2) { // Hadamard product
  const int c = m2.item_size(), r = m2.batch_size();
  assert( item_size()==c );
  assert( batch_size()==r );
  VectorBatch out(r, c, 0);
  for (int i = 0; i < nelements(); i++) {
    out.vals[i] = this->vals[i] / m2.vals[i];
  }
  return out;
}

void VectorBatch::scaleby( float f) {
  for (int i = 0; i < nelements(); i++) {
    vals[i] *= f;
  }
}

VectorBatch VectorBatch::operator-() {
    VectorBatch result = *this;
    for (int i = 0; i < nelements(); i++) {
        result.vals[i] = -vals[i];
    }

    return result;
};

VectorBatch operator/(const VectorBatch &m, const float &c) {
    VectorBatch o = m;
    for (int i = 0; i < m.nelements(); i++) {
        o.vals[i] = o.vals[i] / c;
    }
    return o;
}

VectorBatch operator*(const float &c, const VectorBatch &m) {
    VectorBatch o = m;
    for (int i = 0; i < m.nelements(); i++) {
        o.vals[i] = o.vals[i] * c;
    }
    return o;
}

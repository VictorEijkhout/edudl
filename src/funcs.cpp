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

#include "funcs.h"
#include "trace.h"

#include <iostream>
using std::cout;
using std:: endl;
#include <iomanip>
using std::boolalpha;

#include <algorithm>
#include <numeric>
using std::accumulate;
#include <functional>
using std::function;
#include <vector>
using std::vector;

#include <cmath>
#include <cassert>

//! Batched version of the ReLU function
void relu_io(const VectorBatch &mm, VectorBatch &a) {

  VectorBatch m(mm);
  assert( a.item_size()==m.item_size() );
  assert( a.batch_size()==m.batch_size() );
        auto& avals = a.vals_vector();
  const auto& mvals = m.vals_vector();
  avals.assign(mvals.begin(),mvals.end());
  const float alpha = 0.01; // used for leaky relu, for regular relu, set alpha to 0.0
  for (int i = 0; i < m.batch_size() * m.item_size(); i++) {
    // values will be 0 if negative, and equal to themselves if positive
    if (avals.at(i) < 0)
      avals.at(i) *= alpha;
    //cout << i << ":" << avals.at(i) << endl;
  }
#ifdef DEBUG
  m.display("Apply RELU to");
  a.display("giving");
#endif
}

//codesnippet netsigmoid
//template <typename VectorBatch>
//! Batched version of the sigmoid function
void sigmoid_io(const VectorBatch &m, VectorBatch &a) {

    const auto& mvals = m.vals_vector();
    auto& avals = a.vals_vector();
    avals.assign(mvals.begin(),mvals.end());
    // for (int i = 0; i < m.batch_size() * m.item_size(); i++) {
    //   avals[i] = 1 / (1 + exp(-avals[i]));
    // }
    for ( auto& e: avals ) {
      e = 1.f / ( 1.f + exp( -e ) );
      if (e<1.e-5) e = 1.e-5;
      if (e>1-1.e-5) e = 1-1.e-5;
    }
    if (trace_scalars()) {
      bool limit{true};
      float min{2.f},max{-1.f};
      for_each( avals.begin(),avals.end(),
		[&min,&max,&limit] (auto e) {
		  assert( not isinf(e) ); assert( not isnan(e) );
		  if (e<min) min = e; if (e>max) max = e;
		  limit = limit && e>0 && e<1;
		}
		);
      cout << "sigmoid limited: " << boolalpha << limit
	   << ": " << min << "--" << max << "\n";
      assert( limit );
    }
}
//codesnippet end

//! Batched version of the softmax function
void softmax_io(const VectorBatch &m, VectorBatch &a) {

  const int vector_size = a.item_size(), batch_size = a.batch_size();
  assert( vector_size==m.item_size() );
  assert( batch_size==m.batch_size() );
  vector<float> nB(batch_size,0);
  vector<float> mVectorBatch(batch_size,-99);

  // compute software independently for each vector j
  for (int j = 0; j < batch_size; j++) {
    // copy a vector from m into temporary aj
    auto aj = a.extract_vector(j);
    const auto& mj = m.extract_vector(j);
    std::copy( mj.begin(),mj.end(),aj.begin() );
    // find the max
    mVectorBatch.at(j) = *max_element( aj.begin(),aj.end() );
    // shift down by max to prevent overflow
    for_each( aj.begin(),aj.end(),
              [=] ( auto& x ) { x -= mVectorBatch.at(j); } );
    // exponentiate
    for_each( aj.begin(),aj.end(),
              [] ( auto& x ) { x = exp(x); } );
    // compute sum and normalize
    nB.at(j) = accumulate( aj.begin(),aj.end(),0 );
    if ( nB.at(j)==0 ) throw("softmax aj is zero");
    for_each( aj.begin(),aj.end(),
              [=] ( auto& x ) { x /= nB.at(j); } );
    // limit to (0,1) exclusive
    clip(aj);
    a.set_vector( aj,j );
  }

#ifdef DEBUG
  assert( a.positive() );
  m.display("Apply SoftMAX to");
  a.display("giving");
#endif
}

void clip( vector<float>& aj ) {
  for_each( aj.begin(),aj.end(),
	    [] ( auto& x ) { 
	      if (x <= 1e-7)
		x = 1e-7;
	      if (x >= 1 - 1e-7)
		x = 1 - 1e-7;
	    } );
};

/*!
 * Softmax a single vector.
 * This first converts the vector to a batch, so it is not efficient.
 */
void softmax_vec( const Vector& input, Vector& output ) {
    VectorBatch input_batch(input), output_batch(input_batch);
    softmax_io(input_batch,output_batch);
    output.copy_from( output_batch );
};

//! Identity function
void linear_io(const VectorBatch &m, VectorBatch &a) {
    a.vals_vector().assign(m.vals_vector().begin(),m.vals_vector().end());
}

//! Gradient of the ReLU function
void reluGrad_io(const VectorBatch &m, VectorBatch &a) {
  assert( a.item_size()==m.item_size() );
        auto& avals = a.vals_vector();
  const auto& mvals = m.vals_vector();
  avals.assign(mvals.begin(),mvals.end());
  float alpha = 0.01;
  for ( auto &e : avals) {
    if (e<=0)
      e = alpha;
    else
      e = 1.0;
  }
  if (trace_progress()) {
    assert( a.normf()!=0.f );
    assert( a.notinf() );
    assert( a.notnan() );
  }
}

//! Gradient of the sigmoid function
void sigGrad_io(const VectorBatch &m, VectorBatch &a) {
    assert( m.size()==a.size() );

    const auto& mvals = m.vals_vector();
    auto& avals = a.vals_vector();

    avals.assign(mvals.begin(),mvals.end());
    for ( auto &e : avals )
      e = e * (1.0 - e);
    if (trace_scalars())
      cout << "sigmoid grad " << m.normf() << " => " << a.normf() << "\n";
}

//! Gradient of the softmax \todo implement this
void smaxGrad_io(const VectorBatch &m, VectorBatch &a) {
  assert( m.size()==a.size() );
  throw("Unimplemented smaxGrad_io");
}

/*
 * Los calculation
 */
float logloss_v( const Vector& groundTruth, const Vector& result) {
  assert( groundTruth.size()==result.size() );
  assert( result.positive() );
  const auto& g_values = groundTruth.values();
  const auto& r_values = result.values();
  float sum=0.;
  for ( int i=0; i<result.size(); i++)
    sum += g_values[i]*log(r_values[i]);
  return sum;
};

float meansquare_v( const Vector& groundTruth, const Vector& result) {
  assert( groundTruth.size()==result.size() );
  assert( result.positive() );
  const auto& g_values = groundTruth.values();
  const auto& r_values = result.values();
  float sum=0.;
  for ( int i=0; i<result.size(); i++)
    sum += pow( g_values[i]-r_values[i],2 );
  return sum;
};


#include <limits>
//! Set the output to NaN, used for initialization and unit testing
void nan_io(const VectorBatch &mm, VectorBatch &a) {

  VectorBatch m(mm);
  assert( a.item_size()==m.item_size() );
  assert( a.batch_size()==m.batch_size() );
  auto& avals = a.vals_vector();
  for (int i = 0; i < a.batch_size() * a.item_size(); i++) {
      avals.at(i) = std::numeric_limits<float>::signaling_NaN();
  }
}

//! Gradient of the softmax function
#ifdef USE_GSL
Matrix smaxGrad_vec( const gsl::span<float> &v)
#else
Matrix smaxGrad_vec( const std::vector<float> &v)
#endif
{
	Matrix im(v.size(),1,0); // Input but converted to a matrix

    for (int i=0; i<v.size(); i++){
      float *i_data = im.data();
      *( i_data +i ) // im.mat[i]
	= v[i];
    }

    Matrix dM = im;

    Matrix diag(dM.rowsize(),dM.rowsize(),0);

    for (int i=0,j=0; i<diag.rowsize()*diag.colsize(); i+=diag.rowsize()+1,j++) {
        // identity * dM
      float *d_data = diag.data(), *m_data = dM.data();
      *( d_data+i ) // diag.mat[i]
	= *( m_data+j ); //dM.mat[j];
    }

    // S_(i,j) dot S_(i,k)
    Matrix dMT = dM.transpose();

    Matrix S(dM.rowsize(),dMT.colsize(),0);
	dM.mmp(dMT, S);
    im = diag - S; // Jacobian
    return im;

}


//template <typename VectorBatch>
void linGrad_io(const VectorBatch &m, VectorBatch &a) {
	assert( m.size()==a.size() );
	std::fill(a.vals_vector().begin(), a.vals_vector().end(), 1.0); // gradient of a linear function
}

//! Pointwise version of the identity function, mostly for unit testing
float id_pt( const float& x ) {
    return x;
}

//! Pointwise version of the ReLu function
float relu_pt( const float& x ) {
  float y;
  if (x < 0)
      y = 0.;
  else
      y = x;
  return y;
}

//! Pointwise version of the ReLu function, with leakage for negative input
float relu_slope_pt( const float& x ) {
  const float alpha = 0.01;
  if (x < 0)
      return x * alpha;
  else
      return x;
}

//! Pointwise version of the sigmoid function
float sigmoid_pt( const float& x ) {
    float y = 1.f / ( 1.f + exp( -x ) );
    if (y<1.e-5)
	return 1.e-5;
    else if (y>1-1.e-5)
	return 1-1.e-5;
    else
	return y;
}

/*!
 * If we specify a pointwise function,
 * this computes a batched version of it in the layer
 */
void batch_activation
    ( function< float(const float&) > pointwise,const VectorBatch &i,VectorBatch &o,bool by_vector ) {

  assert( i.item_size()==o.item_size() );
  assert( i.batch_size()==o.batch_size() );
  const auto& ivals = i.vals_vector();
  auto& ovals = o.vals_vector();
  ovals.assign(ivals.begin(),ivals.end());
  if (by_vector) {
    throw("need to think about by-vector activation");
    for ( int v=0; v<o.batch_size(); v++ ) {
      auto vec = o.get_vector(v);
      for_each( vec.begin(),vec.end(), pointwise );
      o.set_vector(vec,v);
    }
  } else {
    for ( size_t i=0; i<o.item_size()*o.batch_size(); i++ ) {
      ovals[i] = pointwise( ivals[i] );
    }
    //    for_each( ovals.begin(),ovals.end(), pointwise );
  }
#ifdef DEBUG
  o.display("Apply pointwise function to");
  i.display("giving");
#endif
  
}


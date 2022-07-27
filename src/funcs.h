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

#ifndef SRC_FUNCS_H
#define SRC_FUNCS_H

#include <functional>

#include "matrix.h"
#include "vector.h"
#include "vector2.h"

#ifdef USE_GSL
#include "gsl/gsl-lite.hpp"
#endif

void relu_io    (const VectorBatch &i, VectorBatch &v);
void sigmoid_io (const VectorBatch &i, VectorBatch &v);
void softmax_io (const VectorBatch &i, VectorBatch &v);
void softmax_vec( const Vector& input, Vector& output );
void linear_io    (const VectorBatch &i, VectorBatch &v);
void nan_io     (const VectorBatch &i, VectorBatch &v);

void reluGrad_io(const VectorBatch &m, VectorBatch &a);
void sigGrad_io (const VectorBatch &m, VectorBatch &a);
void smaxGrad_io(const VectorBatch &m, VectorBatch &a);
void linGrad_io	(const VectorBatch &m, VectorBatch &a);

#ifdef USE_GSL
Matrix smaxGrad_vec( const gsl::span<float> &v);
#else
Matrix smaxGrad_vec( const std::vector<float> &v);
#endif

void clip( std::vector<float>& aj );

enum acFunc{RELU,SIG,SMAX,NONE};

static inline std::vector< std::string > activation_names{
  "ReLU", "Sigmoid", "SoftMax", "Linear" };

template <typename V>
static inline std::vector< std::function< void(const V&, V&) > > apply_activation{
  [] ( const V &v, V &a ) { relu_io(v,a); },
    [] ( const V &v, V &a ) { sigmoid_io(v,a); },
    [] ( const V &v, V &a ) { softmax_io(v,a); },
    [] ( const V &v, V &a ) { linear_io(v,a); }
};
  	
template <typename V>
static inline std::vector< std::function< void(const V&, V&) > > activate_gradient{
  [] (  const V &m, V &v ) { reluGrad_io(m,v); },
    [] (  const V &m, V &v ) { sigGrad_io(m,v); },
    [] (  const V &m, V &v ) { smaxGrad_io(m,v); },
    [] (  const V &m, V &v ) { linGrad_io(m,v); }
};

using lossfunction_t = float( const Vector& groundTruth, const Vector& result);
using lossfunction_ft = std::function< lossfunction_t > ;

float logloss_v( const Vector& groundTruth, const Vector& result);
float meansquare_v( const Vector& groundTruth, const Vector& result);

// enum lossfn{cce, mse}; // categorical cross entropy, mean squared error
// inline static std::vector<
//   std::function< float( const Vector& groundTruth, const Vector& result) > 
//   > lossFunctions{
//   [] ( const float &gT, const float &result ) {
//     assert(result>0.f);
//     return gT * log(result); }, // Categorical Cross Entropy
//   [] ( const float &gT, const float &result ) {
//     return pow(gT - result, 2); }, // Mean Squared Error
// };

inline static std::vector<
  std::function< VectorBatch( VectorBatch&, VectorBatch&) >
  > d_lossFunctions  {
  [] ( VectorBatch &gT, VectorBatch &result ) -> VectorBatch {
    std::cout << "Ambiguous operator\n" ; throw(27);
    //return -gT / ( result/ static_cast<float>( result.batch_size() ) );
  },
  [] ( VectorBatch &gT, VectorBatch &result ) -> VectorBatch {
    std::cout << "Ambiguous operator\n" ; throw(27);
    //return -2 * ( gT-result)/ static_cast<float>( result.batch_size() );
  },
};

float id_pt( const float &x );
float relu_pt( const float &x );
float relu_slope_pt( const float &x );
float sigmoid_pt( const float &x );
void batch_activation
    ( std::function< float(const float&) > pointwise,
      const VectorBatch &i,VectorBatch &o,bool=false );

#include <random>
class Random {
private:
  static inline std::default_random_engine generator{};
public:
  template<typename T,typename V>
  static T random_in_interval(V lo,V hi) {
    std::uniform_real_distribution<T> distribution(lo,hi);
    return distribution(generator);
  };
  static int random_int_in_interval(int lo,int hi) {
    std::uniform_int_distribution distribution(lo,hi);
    return distribution(generator);
  };
};

/*
*/
#endif //SRC_FUNCS_H

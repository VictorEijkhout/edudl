#include <cassert>
#include <numeric>
#include <vector>
using std::vector;

#include "loss.h"

template< typename V >
float Loss<V>::calculate( const V& groundtruth, const V& net_output ) const {
  auto sample_losses = forward(groundtruth,net_output );
  return sample_mean(sample_losses);
};

template< typename V >
float Loss<V>::sample_mean( float v ) const {
  return v;
};  

template< typename V >
float Loss<V>::sample_mean( vector<float> v ) const {
  assert( v.size()>0 );
  auto sum = accumulate( v.begin(),v.end(),
			 static_cast<float>(0) );
  return sum/v.size();
};  

template class Loss< std::vector<float> >;
template class Loss< Vector >;
template class Loss< VectorBatch >;


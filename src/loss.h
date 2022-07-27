// -*- c++ -*_

#include <vector>
#include "vector.h"
#include "vector2.h"

template< typename V >
class Loss {
private:
  std::function< std::vector<float>( const V& t, const V& o ) > forward;
public:
  Loss() {
    forward = [] ( const V& t, const V& o ) -> std::vector<float> {
      throw("no forward function defined"); };
  };
  Loss( std::function< std::vector<float>( const V& t, const V& o ) > forward )
    : forward(forward) {}
  float sample_mean( float ) const;
  float sample_mean( std::vector<float> ) const;
  float calculate( const V& t, const V& o ) const;
};


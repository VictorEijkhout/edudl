
#include <cassert>
#include <tuple>
using std::pair, std::make_pair;
#include "book.h"
#include "funcs.h"
#include "matrix.h"

/*
def create_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) \
	        + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
*/

pair<Matrix,Vector> spiral_data( int samples,int classes ) {
  assert( samples>1 );
  Matrix X( samples*classes,2 );
  Vector y( samples*classes );
  for ( int class_number=0; class_number<classes; class_number++ ) {
    for ( int ix=samples*class_number; ix<samples*(class_number+1); ix++ ) {
      auto r = linspace<float>(samples);
      auto t = linspace<float>( samples, (class_number+1)*4, class_number*4 );
      for ( auto& et : t )
	et += Random::random_in_interval<float>(0.0f, 0.2f);
    }
  }
  return make_pair(X,y);
};

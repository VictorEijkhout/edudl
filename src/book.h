#include <tuple>
#include <vector>

#include "matrix.h"

template<typename T>
std::vector<T> linspace( int n, T hi=static_cast<T>(1), T lo=static_cast<T>(0) ) {
  assert( n>1 );
  std::vector<T> space(n);
  for (int i=0; i<n; i++)
    space[i] = lo + static_cast<T>(i) * (hi-lo) / (n-1);
  return space;
};

std::pair<Matrix,Vector> spiral_data( int samples,int classes );


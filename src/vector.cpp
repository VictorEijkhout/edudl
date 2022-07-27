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

#include <algorithm>
#include <iostream>
#include <cassert>

#include "vector.h"
#include "vector2.h"

/*
 * These are the vector routines that do not have a optimized implementation,
 * such as using BLIS.
 */


Vector::Vector(){
}

int Vector::size() const { return vals.size(); };

Vector::Vector( std::vector<float> vals )
  : vals(vals) {
};

void Vector::copy_from( const VectorBatch& batch ) {
    const auto& batch_vals = batch.data();
    assert( batch.nelements()>=vals.size() );
    for (int i=0; i<vals.size(); i++) {
	vals[i] = batch_vals[i];
    }
};

void Vector::square() {
    std::for_each(vals.begin(), vals.end(), [](auto &n) {n*=n;});
}

Vector& Vector::operator=(const Vector &m2) { // Overloading the = operator
	vals = m2.vals;
    return *this;
}

// Note: no element-wise, non destructive operations in BLIS, so no implementations for those yet
// There are element wise operations in MKL I believe
Vector Vector::operator+(const Vector &m2) {
    assert(m2.size()==this->size());
    Vector out(m2.size(),0);
    for (int i=0;i<m2.size();i++) {
        out.vals[i] = this->vals[i] + m2.vals[i];
    }
    return out;
}

Vector Vector::operator-(const Vector &m2) {
    assert(m2.size()==this->size());
    Vector out(m2.size(),0);
    for (int i=0;i<m2.size();i++) {
        out.vals[i] = this->vals[i] - m2.vals[i];
    }
    return out;
}


Vector operator-(const float &c, const Vector &m) {
    Vector o=m;
    for (int i=0;i<m.size();i++) {
        o.vals[i] = c - o.vals[i];
    }
    return o;
}

Vector operator*(const float &c, const Vector &m) {
    Vector o=m;
    for (int i=0;i<m.size();i++) {
        o.vals[i] = c * o.vals[i];
    }
    return o;
}

Vector Vector::operator/=( float c ) {
    for (int i=0;i<vals.size();i++) {
        vals[i] = vals[i] / c;
    }
    return *this;
}

Vector Vector::operator*(const Vector &m2) { // Hadamard product
    Vector out(m2.size(), 0);
    for (int i = 0; i < m2.size(); i++) {
        out.vals[i] = this->vals[i] * m2.vals[i];
    }
    return out;
}

Vector Vector::operator/(const Vector &m2) { // Element wise division
    Vector out(m2.size(), 0);
    for (int i = 0; i < m2.size(); i++) {
        out.vals[i] = this->vals[i] / m2.vals[i];
    }
    return out;
}

Vector Vector::operator-() {
    Vector result = *this;
    for (int i = 0; i < size(); i++){
        result.vals[i] = -vals[i];
    }

    return result;
};

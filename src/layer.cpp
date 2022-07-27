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

#include "layer.h"
#include "trace.h"

#include <iostream>
using std::cout;
using std::endl;

Layer::Layer() {};
Layer::Layer(int insize,int outsize)
  : weights( Matrix(outsize,insize,1) ),
    dw     ( Matrix(outsize,insize, 0) ),
	//dW( Matrix(outsize,insize, 0) ),
    dw_velocity( Matrix(outsize,insize, 0) ),
    biases( Vector(outsize, 1 ) ),
    // biased_product( Vector(outsize, 0) ),
    activated( Vector(outsize, 0) ),
    d_activated( Vector(outsize, 0) ),
    delta( VectorBatch(outsize,1) ),
    // biased_productm( VectorBatch(outsize,insize,0) ),
    //    activated_batch( VectorBatch(outsize,1, 0) ),
    //    d_activated_batch ( VectorBatch(outsize,insize, 0) ),
    db( Vector(insize, 0) ),
    // delta_mean( Vector(insize, 0) ),
    dl( VectorBatch(insize, 1) ),
    db_velocity( Vector(insize, 0) ) {};

/*
 * Resize temporaries to reflect current batch size
 */
void Layer::allocate_batch_specific_temporaries(int batchsize) {
  const int insize = weights.colsize(), outsize = weights.rowsize();

  biased_batch.allocate( batchsize,outsize );
  input_batch.allocate( batchsize,insize );
  //  activated_batch.allocate( batchsize,outsize );
  d_activated_batch.allocate( batchsize,outsize );
  dl.allocate( batchsize, outsize );
  delta.allocate( batchsize,outsize );
};

void Layer::set_activation(acFunc f) {
  activation = f;
  apply_activation_batch  = apply_activation<VectorBatch>.at(f);
  activate_gradient_batch = activate_gradient<VectorBatch>.at(f);
};

Layer& Layer::set_activation
    (
     std::function< float(const float&) > activate_pt,
     std::function< float(const float&) > gradient_pt,
     std::string name
     ) {
    set_activation
	(
	 [activate_pt] ( const VectorBatch& i,VectorBatch& o ) -> void {
	     batch_activation( activate_pt,i,o ); },
	 [gradient_pt] ( const VectorBatch& i,VectorBatch& o ) -> void {
	     batch_activation( gradient_pt,i,o ); },
	 name );
  return *this;
};

Layer& Layer::set_activation
    ( std::function< void(const VectorBatch&,VectorBatch&) > apply,
      std::function< void(const VectorBatch&,VectorBatch&) > activate,
      std::string name ) {
  activation = acFunc::RELU;
  apply_activation_batch  = apply;
  activate_gradient_batch = activate;
  return *this;
};

Layer& Layer::set_uniform_weights(float v) {
  for ( auto& e : weights.values() )
    e = v;
  return *this;
};

Layer& Layer::set_uniform_biases(float v) {
  for ( auto& e : biases.values() )
    e = v;
  return *this;
};

//codesnippet layerforward
void Layer::forward( const VectorBatch& input, VectorBatch& output) {
  assert( input.batch_size()==output.batch_size() );
  if (trace_progress()) {
    cout << "Forward layer " << layer_number
	 << ": " << input_size() << "->" << output_size() << endl;
  }

  allocate_batch_specific_temporaries(input.batch_size());
  input_batch.copy_from(input);
  if (trace_progress()) {
    assert( input_batch.notnan() ); assert( input_batch.notinf() );
    assert( weights.notnan() ); assert( weights.notinf() );
  }
  input_batch.v2mp( weights, biased_batch );
  if (trace_progress()) {
    assert( biased_batch.notnan() ); assert( biased_batch.notinf() );
  }
  
  biased_batch.addh(biases); // Add the bias
  if (trace_progress()) {
    assert( biased_batch.notnan() ); assert( biased_batch.notinf() );
  }
  
  apply_activation_batch(biased_batch, output);
  //cout << "layer output: " << output.data()[0] << "\n";
  if (trace_progress()) {
    assert( output.notnan() ); assert( output.notinf() );
  }
  //return activated_batch;
}
//codesnippet end

const VectorBatch& Layer::intermediate() const { return biased_batch; };

void Layer::backward
    (const VectorBatch &prev_delta, const Matrix &W, const VectorBatch &prev_output) {

  // compute delta ell
  activate_gradient_batch(prev_output, d_activated_batch); 
  prev_delta.v2mtp( W, dl );

   // delta  = Dl . sigma
  if (trace_progress())
    cout << "L-" << layer_number << " delta\n";
  delta.hadamard( d_activated_batch,dl ); // Derivative of the current layer

  // prev_output.outer2( delta, dw );
  // if (trace_scalars())
  //   cout << "L-" << layer_number << " dw: "
  // 	 << delta.normf() << "x" << prev_output.normf() << " => " << dw.normf() << "\n";

  update_dw(delta, prev_output);
  // weights.axpy( 1.,dw );
  // db = delta.meanh();
  // biases.add( db );
}

void Layer::update_dw( const VectorBatch &delta, const VectorBatch& prev_output) {
   prev_output.outer2( delta, dw );
   if (trace_scalars())
     cout << "L-" << layer_number << " dw: "
	  << delta.normf() << "x" << prev_output.normf() << " => " << dw.normf() << "\n";

  // Delta W = delta here X activated prevous
  weights.axpy( 1.,dw );
  db = delta.meanh();
  biases.add( db );
}

void Layer::set_topdelta( const VectorBatch& gTruth,const VectorBatch& output ) {

    // top delta ell is different
   activate_gradient_batch(output, d_activated_batch); 
   dl = output - gTruth;
   dl.scaleby( 1.f / gTruth.batch_size() );
   // delta  = Dl . sigma
   delta.hadamard( d_activated_batch,dl );
   if (trace_scalars())
     cout << "L-" << layer_number << " top delta: "
	  << d_activated_batch.normf() << "x" << dl.normf() << " => " << delta.normf() << "\n";

   //  update_dw(delta, prev_output);
};

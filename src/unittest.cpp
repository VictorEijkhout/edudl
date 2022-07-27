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
#include <vector>
using std::vector;

#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "book.h"
#include "funcs.h"
#include "loss.h"
#include "net.h"

/****************************************************************
 **** Auxiliary stuff
 ****************************************************************/

TEST_CASE( "auxiliaries","[0]" ) {
  vector<float> tenths;
  float lo=0.f, hi=1.f;
  SECTION( "default 0--1" ) {
    REQUIRE_NOTHROW( tenths = linspace<float>(11) );
  }
  SECTION( "stretch 0--2" ) {
    hi = 2.f;
    REQUIRE_NOTHROW( tenths = linspace<float>(11,hi) );
  }
  SECTION( "way out" ) {
    lo = 10.f; hi = 200.f;
    REQUIRE_NOTHROW( tenths = linspace<float>(11,hi,lo) );
  }
  INFO( "lo=" << lo << ", hi=" << hi );
  REQUIRE( tenths[0]==lo );
  REQUIRE( tenths[1]==Catch::Approx( lo + .1f * (hi-lo) ) );
  REQUIRE( tenths.back()==Catch::Approx(hi) );
}

/****************************************************************
 **** Activation functions
 ****************************************************************/
TEST_CASE( "functions","[1]" ) {
  /*
   * ReLU is linear >0, zero <0
   */
  {
    auto highpass = [] ( const float &x ) -> float {
      return relu_pt(x); };
    auto x = GENERATE( -5., -.5, .5, 2.5 );
    float y;
    REQUIRE_NOTHROW( y = highpass(x) );
    if (x<0)
      REQUIRE( y==Catch::Approx(0.) );
    else
      REQUIRE( y==Catch::Approx(x) );
  }

  /*
   * relu_slope avoids zero'ing the negative inputs
   */
  {
    auto leak = [] ( const float &x ) -> float {
      return relu_slope_pt(x); };
    auto x = GENERATE( -5., -.5, .5, 2.5 );
    float y;
    REQUIRE_NOTHROW( y = leak(x) );
    if (x<0)
      REQUIRE( y<0 );
    else
      REQUIRE( y==Catch::Approx(x) );
  }
}

TEST_CASE( "batch activation","[02]" ) {
  Vector values( vector<float>{1,2,.5} ), maxes(3);
  REQUIRE_NOTHROW( softmax_vec( values,maxes ) );
  REQUIRE( maxes.positive() );
  const auto& maxvalues = maxes.values();
  INFO( "maxes: " << maxvalues[0] << "," << maxvalues[1] << "," << maxvalues[2] );
  auto maxit = maxvalues.begin();
  REQUIRE_NOTHROW( maxit = find( maxvalues.begin(),maxvalues.end(),
				 *max_element( maxvalues.begin(),maxvalues.end() ) ) );
  REQUIRE( maxit!=maxvalues.end() );
  int maxloc{-1};
  REQUIRE_NOTHROW( maxloc = distance( maxvalues.begin(),maxit ) );
  REQUIRE( maxloc==1 );
}

/****************************************************************
 **** Data tools
 ****************************************************************/

TEST_CASE( "Categorization","[categorization][11]" ) {
  Categorization zero( vector<float>{ .0, .0 } );
  REQUIRE_THROWS( zero.normalize() );

  Categorization cat( { .15, .2, .4, .3 } );
  INFO( cat.as_string() );
  REQUIRE_NOTHROW( cat.normalize() );
  int m=-1;
  REQUIRE_NOTHROW( m = cat.onehot() );
  REQUIRE( m==2 );
}

/****************************************************************
 **** Loss stuff
 ****************************************************************/
TEST_CASE( "Loss calculation","[loss][21]" ) {
  Loss< vector<float> > calc;
  float mean;
  REQUIRE_NOTHROW( mean = calc.sample_mean( .3 ) );
  REQUIRE( mean==Catch::Approx( .3 ) );
  REQUIRE_NOTHROW( mean = calc.sample_mean( vector<float>{.1f,.2f,.6f} ) );
  REQUIRE( mean==Catch::Approx( .3 ) );
}


/****************************************************************
 **** Layer and network test
 ****************************************************************/

TEST_CASE( "scalar layer","[layer][51]" ) {
    Layer multiply(1,1);
    REQUIRE_NOTHROW
	( 
	 multiply
	 .set_uniform_weights(1.)
	 .set_uniform_biases(0.)
	 .set_activation(
			 [] (const float &x ) -> float { return id_pt(x); },
			 [] (const float &x ) -> float { return 1; },
			 "id"
			 )
	  );
    float input_number = 2.5f;
    VectorBatch
	scalar_in( Vector( vector<float>( {input_number} ) ) ),
	scalar_out( Vector( vector<float>( {1.2f} ) ) );
    REQUIRE( scalar_in.batch_size()==1 );
    REQUIRE( scalar_in.item_size()==1 );
    REQUIRE( scalar_out.batch_size()==1 );
    REQUIRE( scalar_out.item_size()==1 );
    REQUIRE( multiply.input_size()==1 );
    REQUIRE( multiply.output_size()==1 );
    float correct_result;
    SECTION( "identity" ) {
	correct_result = input_number;
    }
    SECTION( "scale by 2" ) {
	float multiplier = 2.f;
	REQUIRE_NOTHROW( multiply.set_uniform_weights(multiplier) );
	correct_result = multiplier * input_number;
    }
    SECTION( "shift by 2" ) {
	float shift = 2.f;
	REQUIRE_NOTHROW( multiply.set_uniform_biases(shift) );
	correct_result = shift + input_number;
    }
    REQUIRE_NOTHROW( multiply.forward(scalar_in,scalar_out) );
    REQUIRE( scalar_out.at(0,0) ==Catch::Approx( correct_result ) );    
}

TEST_CASE( "highpass net: batch version","[net][61]" ) {
  Net highpass(1);
  REQUIRE_NOTHROW( highpass.addLayer
		   (1,
		    [] (const float &x ) -> float { return relu_pt(x); },
		    [] (const float &x ) -> float { return x; },
		    "highpass" ) );
  REQUIRE_NOTHROW( highpass.backlayer()
		   .set_uniform_weights(1.)
		   .set_uniform_biases(0.)
		   );
  VectorBatch input(1,1),output(1,1);
  auto x = GENERATE( -.5, -5, .5, 1.5 );
  input.data()[0] = x;
  REQUIRE_NOTHROW( highpass.feedForward(input,output) );
  auto y = output.data()[0];
  INFO( "Input = " << x << "-> output = " << y );
  if ( x>0 )
      REQUIRE( y>0 );
  else
      REQUIRE( y==Catch::Approx(0.0) );
}

TEST_CASE( "highpass net: vector version","[net][62]" ) {
  Net highpass(1);
  REQUIRE_NOTHROW( highpass.addLayer
		   (1,
		    [] (const float &x ) -> float { return relu_pt(x); },
		    [] (const float &x ) -> float { return x; },
		    "highpass" ) );
  REQUIRE_NOTHROW( highpass.backlayer()
		   .set_uniform_weights(1.)
		   .set_uniform_biases(0.)
		   );
  Vector input(1),output(1);
  auto x = GENERATE( -.5, -5, .5, 1.5 );
  input[0] = x;
  REQUIRE_NOTHROW( highpass.feedForward(input,output) );
  INFO( "Input = " << input[0] << "-> output = " << output[0] );
  if ( x>0 )
      REQUIRE( output[0]>0 );
  else
      REQUIRE( output[0]==Catch::Approx(0.0) );
}

TEST_CASE( "low pass net: vector version","[net][63]" ) {
  Net lowpass(1);
  REQUIRE_NOTHROW( lowpass.addLayer
		   (1,
		    [] (const float &x ) -> float { return relu_pt(1-x); },
		    [] (const float &x ) -> float { return x; },
		    "lowpass" ) );
  REQUIRE_NOTHROW( lowpass.backlayer()
		   .set_uniform_weights(1.)
		   .set_uniform_biases(0.)
		   );
  Vector input(1),output(1);
  auto x = GENERATE( -.5, -5, .5, 1.5 );
  input[0] = x;
  REQUIRE_NOTHROW( lowpass.feedForward(input,output) );
  INFO( "Input = " << input[0] << "-> output = " << output[0] );
  if ( x<1 )
      REQUIRE( output[0]>0 );
  else
      REQUIRE( output[0]==Catch::Approx(0.0) );
}

TEST_CASE( "create multilayer network","[net][64]" ) {
  Net twolayers(2);
  REQUIRE_NOTHROW( twolayers.addLayer(4,relu_io,reluGrad_io) );
  REQUIRE_NOTHROW( twolayers.addLayer(2,softmax_io,smaxGrad_io) );
  
}

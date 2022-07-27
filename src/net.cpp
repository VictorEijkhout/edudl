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
#include <fstream>

#include <algorithm>
#include <string>
using std::string;

#include <cmath>

#include "vector.h"
#include "net.h"
#include "trace.h"

/*!
  Create a network with a given input number of features
*/
Net::Net(int s) { // Input vector size
    this->inR = s;
}

/*!
  Create a network based on a dataset.
  We do not store the dataset.
*/
Net::Net( const Dataset &data )
  : Net(data.data_size()) {};

/*!
 * Create a layer from pointwise activation functions
 */
void Net::addLayer( int outputsize,
		    std::function< float(const float&) > activate_pt,
		    std::function< float(const float&) > gradient_pt,
		    string name ) {
    addLayer
	( outputsize,
	  [activate_pt] ( const VectorBatch& i,VectorBatch& o ) -> void {
	      batch_activation( activate_pt,i,o ); },
	  [gradient_pt] ( const VectorBatch& i,VectorBatch& o ) -> void {
	      batch_activation( gradient_pt,i,o ); },
	  name );
};

/*!
 * Create a layer from vector activation functions
 */
void Net::addLayer( int outputsize,
		    std::function< void(const VectorBatch&,VectorBatch&) > apply_activation_batch,
		    std::function< void(const VectorBatch&,VectorBatch&) > activate_gradient_batch,
		    string name
		    ) {
    // Initialize layer object and add the necessary parameters
    Layer layer(inputsize(layers.size()), outputsize);
    layer.set_activation(apply_activation_batch,activate_gradient_batch);
    layer.set_number( layers.size() );
    layer.activation_name = name;
    push_layer(layer);
};

/*!
 * Push a layer as last layer of a network
 */
void Net::push_layer( const Layer& layer ) {
    if (trace_progress())
	cout << "Creating layer " << layers.size()
	     << " of size " << layer.output_size() << "x" << layer.input_size()
	    //	     << " with " << name << " activation"
	     << endl;
    layers.push_back(layer);
};

/*!
 * Add a layer from indexed activation function
 \todo this one probably has to go
*/
void Net::addLayer(int l, acFunc f) {
  addLayer( l,
	    apply_activation<VectorBatch>.at(f),
	    activate_gradient<VectorBatch>.at(f),
	    activation_names.at(f)
	    );
}

// /*!
//  * Set an indexed loss function
//  */
// void Net::set_lossfunction( lossfn lossFuncName ) {
//   lossFunction = lossFunctions.at(lossFuncName);
//   d_lossFunction = d_lossFunctions.at(lossFuncName);
// };

/*!
 * Initialize weights of all layers to a uniform value
 */
void Net::set_uniform_weights(float v) {
  for ( auto& l : layers )
    l.set_uniform_weights(v);
};

/*!
 * Initialize biases of all layers to a uniform value
 */
void Net::set_uniform_biases(float v) {
  for ( auto& l : layers )
    l.set_uniform_biases(v);
};

/*!
 * Feed an input forward through the network
 * The input here is a batch of vectors
 */
//codesnippet netforward
void Net::feedForward( const VectorBatch& input,VectorBatch& output ) {
  if (trace_progress())
    cout << "Feed forward batch of size " << input.batch_size() << endl;
  allocate_batch_specific_temporaries(input.batch_size());

  if (layers.size()==1) {
    layers.front().forward(input,output);
    //cout << "single layer output: " << output.data()[0] << "\n";
  } else {
    layers.front().forward( input, layers.at(1).input() );
    cout << " first layer : "
	 << input.data()[0] << " -> " << layers.at(1).input().data()[0] << "\n";
    for (unsigned i = 1; i<layers.size()-1; i++) {
      layers.at(i).forward
	(layers.at(i).input(),
	 layers.at(i+1).input()
	 );
    }
    layers.back().forward(layers.back().input(),output);
    cout << "last layer : "
	 << layers.back().input().data()[0] << " -> " << output.data()[0] << "\n";
  }
};
//codesnippet end

//! Feed forward with allocated result
Vector Net::feedForward( const Vector& input ) {
  Vector result(input);
  feedForward(input,result);
  return result;
};

//! Feed forward with allocated result
VectorBatch Net::feedForward( const VectorBatch& input ) {
  VectorBatch result(input);
  feedForward(input,result);
  return result;
}

/*!
 * Feed a single vector input forward through the network
 * This first converts the vector to a batch, so it is not efficient.
 */
void Net::feedForward( const Vector& input, Vector& output ) {
    VectorBatch input_batch(input), output_batch(input_batch);
    feedForward(input_batch,output_batch);
    const auto& in_data = input_batch.data();
    const auto& out_data = output_batch.data();
    //    cout << "net forward: " << in_data[0] << " -> " << out_data[0] << "\n";
    output.copy_from( output_batch );
};

/*!
 * Show the weights of all layers
 */
void Net::show() {
    for (unsigned i = 0; i < layers.size(); i++) {
        cout << "Layer " << i << " weights" << endl;
        layers.at(i).weights.show();
    }
}

/*!
 * Set up delta values for back propagation
 * \todo stuff to be done?
 */
void Net::calculate_initial_delta(VectorBatch &input, VectorBatch &gTruth) {
	VectorBatch d_loss = d_lossFunction( gTruth, input);

	if (layers.back().activation_name == "SoftMax" ) { // Softmax derivative function
		Matrix jacobian( input.item_size(), input.item_size(), 0 );
		for(int i = 0; i < input.batch_size(); i++ ) {
		  auto one_column = input.get_vector(i);
		  jacobian = smaxGrad_vec( one_column );
		  Vector one_vector( jacobian.rowsize(), 0 );
		  Vector one_grad = d_loss.get_vectorObj(i);
		  jacobian.mvp( one_grad, one_vector );
		  layers.back().d_activated_batch.set_vector(one_vector,i);
		}
	}
	/* Will add the rest of the code here, not done yet
	*/
}

/*!
 * Full back propagation sweep
 */
void Net::backPropagate
    (const VectorBatch &input, const VectorBatch& output, const VectorBatch &gTruth) {

  if (layers.size()==1) {
    throw(string("single layer case does not work"));
      // const VectorBatch& prev = input;
      // layers.back().update_dw(delta, prev);
      // return;
  } else {

    if (trace_progress()) cout << "Layer-" << layers.back().layer_number << "\n";
    layers.back().set_topdelta( gTruth,output );
    const VectorBatch& prev = layers.back().input();
    layers.back().update_dw(layers.back().delta, prev);


    for (unsigned i = layers.size() - 2; i > 0; i--) {
      if (trace_progress()) cout << "Layer-" << layers.at(i).layer_number << "\n";
      layers.at(i).backward
	  ( layers.at(i+1).delta, layers.at(i+1).weights, layers.at(i).input());
    }

    if (trace_progress()) cout << "Layer-" << layers.at(0).layer_number << "\n";
    layers.at(0).backward(layers.at(1).delta, layers.at(1).weights, input);
	
  }
}

/*!
 * Stochastic Gradient Descent algorithm
 */
void Net::SGD(float lr, float momentum) {
    assert( layers.size()>0 );
    int samplesize = layers.front().input().batch_size();
    for (int i = 0; i < layers.size(); i++) {
        // Normalize gradients to avoid exploding gradients
	Matrix deltaW = layers.at(i).dw / static_cast<float>(samplesize);
        Vector deltaB = layers.at(i).db / static_cast<float>(samplesize);

        // Gradient descent
        if (momentum > 0.0) {
            layers.at(i).dw_velocity = momentum * layers.at(i).dw_velocity - lr * deltaW;
            //layers.at(i).weights = layers.at(i).weights + layers.at(i).dw_velocity;
	    layers.at(i).weights.axpy( 1.f,layers.at(i).dw_velocity );
        } else {
	  //layers.at(i).weights = layers.at(i).weights - lr * deltaW;
	  layers.at(i).weights.axpy( -lr,deltaW );
        }

        layers.at(i).biases = layers.at(i).biases - lr * deltaB;

        // Reset the values of delta sums
        layers.at(i).dw.zeros();
        layers.at(i).db.zeros();
    }
}

void Net::RMSprop(float lr, float momentum) {
    for (int i = 0; i < layers.size(); i++) {
        // Get average over all the gradients
        Matrix deltaWsq = layers.at(i).dw;
        Vector deltaBsq = layers.at(i).db;
       	
	// Gradient step
        deltaWsq.square(); // dW^2
        deltaBsq.square(); // db^2
        // Sdw := m*Sdw + (1-m) * dW^2
		
	layers.at(i).dw_velocity = momentum * layers.at(i).dw_velocity + (1 - momentum) * deltaWsq;
        layers.at(i).db_velocity = momentum * layers.at(i).db_velocity + (1 - momentum) * deltaBsq;
		
	Matrix sqrtSdw = layers.at(i).dw_velocity;
        std::for_each(sqrtSdw.values().begin(), sqrtSdw.values().end(),
		      [](auto &n) { 
			n = sqrt(n);
			if(n==0) n= 1-1e-7;
		      });
	Vector sqrtSdb = layers.at(i).db_velocity;
        std::for_each(sqrtSdb.values().begin(), sqrtSdb.values().end(),
		      [](auto &n) { 
			n = sqrt(n);
			if(n==0) n= 1-1e-7;
		      });
		
        // W := W - lr * dW / sqrt(Sdw)
        layers.at(i).weights = layers.at(i).weights - lr * layers.at(i).dw / sqrtSdw;
        layers.at(i).biases = layers.at(i).biases - lr * layers.at(i).db / sqrtSdb;
			
        // Reset the values of delta sums
        layers.at(i).dw.zeros();
        layers.at(i).db.zeros();
    }
}

// this function no longer used
// void Net::calcGrad(VectorBatch data, VectorBatch labels) {
//     feedForward(data);
//     backPropagate(data, labels);
// }


void Net::train( const Dataset &train_data,const Dataset &test_data,
		 int epochs, int batchSize ) {

    const int Optimizer = optimizer();
    cout << "Optimizing with ";
    switch (Optimizer) {
    case sgd:  cout << "Stochastic Gradient Descent\n";  break;
    case rms:  cout << "RMSprop\n"; break;
    }
	
    std::vector<Dataset> batches = train_data.batch(batchSize);
    float lrInit = learning_rate();
    const float momentum_value = momentum();

    for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
      // Iterate through the entire dataset for each epoch
      cout << endl << "Epoch " << i_epoch+1 << "/" << epochs << endl;
      float current_learning_rate = lrInit; // Reset the learning rate to undo decay

      for (int j = 0; j < batches.size(); j++) {
	// Iterate through all batches within dataset
	auto& batch = batches.at(j);
#ifdef DEBUG
	cout << ".. batch " << j << "/" << batches.size() << " of size " << batch.size() << "\n";
#endif
	//	allocate_batch_specific_temporaries(batch.size());
	VectorBatch batch_output( batch.inputs().item_size(),batch.size() );
	feedForward(batch.inputs(),batch_output);
	backPropagate(batch.inputs(),batch.labels(),batch_output);

        // User chosen optimizer
        current_learning_rate = current_learning_rate / (1 + decay() * j);
        optimize.at(Optimizer)(current_learning_rate, momentum_value); 
		
      }
      auto loss = DatasetLoss(test_data);
      cout << " Loss: " << loss << endl;
      auto acc = accuracy(test_data);
      cout << " Accuracy on trest set: " << acc << endl;
    }

}

/*!
 * Resize temporaries to reflect current batch size
 */
void Net::allocate_batch_specific_temporaries(int batchsize) {
#ifdef DEBUG
  cout << "allocating temporaries for batch size " << batchsize << endl;
#endif
  for ( auto& layer : layers )
    layer.allocate_batch_specific_temporaries(batchsize);
}

/*!
 * Calculate the los function as sum of losses
 * of the individual data point.
 */
//codesnippet netloss
float Net::DatasetLoss(const Dataset &testSplit) {

#ifdef DEBUG
  cout << "Loss calculation\n";
#endif

  /*
  VectorBatch result = this->create_output_batch( testSplit.inputs().batch_size() );
  feedForward( testSplit.inputs(),result );
  */
  auto result = feedForward( testSplit.inputs() );
  assert( result.notnan() );

  return BatchLoss( testSplit.labels(),result );
}

float Net::BatchLoss( const VectorBatch& labels,const VectorBatch& net_output ) {
  float loss = 0.0;
  assert( labels.notnan() ); assert( net_output.notnan() );
  auto tmp_labels = labels;
  if (trace_arrays()) {
    cout << "Compare net_outputs\n"; net_output.show();
    cout << " to label\n"; tmp_labels.show();
  }
  const int batchsize = net_output.batch_size();
  assert( batchsize>0 );
  for (int vec=0; vec<batchsize; vec++) { // iterate over all items
    const auto& one_result = net_output.get_vector(vec); // VLE figure out const span !!!
    const auto& one_label  = tmp_labels.get_vector(vec);
    assert( one_result.size()==one_label.size() );
    loss += lossFunction( one_result,one_label );
  }
  auto scale = 1.f / static_cast<float>(batchsize);
  loss = loss * scale;
    
  return loss;
};
//codesnippet end

float Net::BatchLoss( const Vector& labels,const Vector& net_output ) {
  VectorBatch label_batch(labels), output_batch(net_output);
  return BatchLoss( label_batch,output_batch );  
};

float Net::accuracy( const Dataset &test_set ) {
  if (trace_progress())
    cout << "Accuracy calculation\n";

  int correct = 0;
  int incorrect = 0;

  assert( test_set.size()>0 );
  const auto& test_inputs = test_set.inputs();
  const auto& test_labels = test_set.labels();
  assert( test_inputs.batch_size()==test_labels.batch_size() );
  assert( test_inputs.batch_size()>0 );

  //      allocate_batch_specific_temporaries(test_inputs.batch_size());
  if (trace_arrays()) {
      cout << "inputs:\n"; test_inputs.show();
  }
  assert( test_inputs.normf()!=0.f );
  auto output_batch = this->create_output_batch(test_labels.batch_size());
  feedForward(test_inputs,output_batch);
  if (trace_arrays()) {
      cout << "outputs:\n"; output_batch.show();
  }
  assert( output_batch.notnan() );

  for(int idx=0; idx < output_batch.batch_size(); idx++ ) {
      Vector oneItem = output_batch.get_vectorObj(idx);
      Categorization result( oneItem );
      result.normalize();
      if ( result.close_enough( test_labels.extract_vector(idx) ) ) {
	  correct++;
      } else {
	  incorrect++;
      }
  }
  assert( correct+incorrect==test_set.size() );

  float acc = static_cast<float>( correct ) / static_cast<float>( test_set.size() );
  return acc;
}



/*!
 * Write the model to file
 */
void Net::saveModel(std::string path){
	/*
	1. Size of matrix m x n, activation function
	2. Values of weight matrix
	3. Values of bias vector 
	4. Repeat for all layers
	*/
	std::ofstream file;
	file.open( path, std::ios::binary );
	
	float temp;
	int no_layers = layers.size();
	file.write( reinterpret_cast<char *>(&no_layers), sizeof(no_layers) ); 
	for ( auto l : layers ) {
		int insize = l.input_size(), outsize = l.output_size();
		file.write(	reinterpret_cast<char *>(&outsize), sizeof(int) );
		file.write(	reinterpret_cast<char *>(&insize), sizeof(int) );
		file.write( reinterpret_cast<char *>(&l.activation), sizeof(int) );
		
		const auto& weights = l.weights;
		file.write(reinterpret_cast<const char *>(weights.data()), sizeof(float)*weights.nelements());
		// const float* weights_data = l.weights.data();
		// file.write(reinterpret_cast<char *>(&weights_data), sizeof(temp)*insize*outsize);

		const auto& biases = l.biases;
		file.write(reinterpret_cast<const char*>(biases.data()), sizeof(float) * biases.size());
		//file.write(reinterpret_cast<char *>(&l.biases.vals[0]), sizeof(temp) * l.biases.size());
	}
	cout << endl;

	file.close();
}

void Net::loadModel(std::string path){
	std::ifstream file(path);
	std::string buffer;
	
	int no_layers;
	file.read( reinterpret_cast<char *>(&no_layers), sizeof(no_layers) );
	
	float temp;
	layers.resize(no_layers);
	for ( int i=0; i < layers.size(); i++ ) {
		int insize,outsize;
		file.read( reinterpret_cast<char *>(&outsize), sizeof(int) );
		file.read( reinterpret_cast<char *>(&insize), sizeof(int) );
	
		file.read( reinterpret_cast<char *>(&layers[i].activation), sizeof(int) );

		layers[i].weights = Matrix( outsize, insize, 0 );
		float *w_data = layers[i].weights.data();
		file.read(reinterpret_cast<char *>( w_data ), //(&layers[i].weights.mat[0]), 
			  sizeof(temp) * insize*outsize);

		layers[i].biases = Vector( outsize, 0 );
		float *b_data = layers[i].biases.data();
		file.read(reinterpret_cast<char *>( b_data ), //(&layers[i].biases.vals[0]), 
			  sizeof(temp) * layers[i].biases.size());
	}
}


void Net::info() {
	cout << "Model info\n---------------\n";

	for ( auto l : layers ) {
		cout << "Weights: " << l.output_size() << " x " << l.input_size() << "\n";
		cout << "Biases: " << l.biases.size() << "\n";
		cout << "Activation: " << l.activation_name << "\n";
		
		// switch (l.activation) {
		// 	case RELU: cout << "RELU\n"; break;
		// 	case SIG: cout << "Sigmoid\n"; break;
		// 	case SMAX: cout << "Softmax\n"; break;
		// 	case NONE: break;
		// }
		cout << "---------------\n";
	}

}

void loadingBar(int currBatch, int batchNo, float acc, float loss) {
	cout  << "[";
   	int pos = 50 * currBatch/(batchNo-1);
   	for (int k=0; k < 50; ++k) {
    if (k < pos) cout << "=";
    else if (k == pos) cout << ">";
    else cout << " ";
    }
    cout << "] " << int(float(currBatch)/float(batchNo-1)*100) << "% " << "loss: " << loss << " acc: " << acc << " \r";
	cout << std::flush;
}

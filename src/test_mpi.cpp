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
#include <chrono>
#include <vector>
#include <time.h>

#include "cxxopts.hpp"

#include "net.h"
#include "dataset.h"

#include <mpl/mpl.hpp>
using namespace std;

#if 0
int main(){
	srand(time(NULL));

	Net bar(0);
	bar.loadModel("weights.bin");	
	bar.info();	

    return 0;
}
#else
int main(int argc,char **argv){
    using myclock = std::chrono::high_resolution_clock;
    srand(time(NULL));

	const mpl::communicator &comm_world(mpl::environment::comm_world());	
    // IM attempt to use cxxopts to specify location
    cxxopts::Options options("EduDL", "FFNNs w BLIS");

    options.add_options()
      ("h,help","usage information")
      ("d,dir", "Dataset directory",cxxopts::value<std::string>())
      ("l,levels", "Number of levels in the network",cxxopts::value<int>()->default_value("2"))
      ("s,sizes","Sizes of the levels",cxxopts::value<std::vector<int>>())
      ("o,optimizer", "Optimizer to be used, 0: SGD, 1: RMSprop",cxxopts::value<int>()->default_value("0"))
	  ("e,epochs", "Number of epochs to train the network", cxxopts::value<int>()->default_value("1"))
	  ("r,learningrate", "Learning rate for the optimizer", cxxopts::value<float>()->default_value("0.001"))
	  ("b,batchsize", "Batch size for the training data", cxxopts::value<int>()->default_value("256"))
	  ;
		
    auto result = options.parse(argc,argv);
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 1;
    }

    std::vector<int> level_sizes{12,12};

    if (result.count("sizes")) {
      level_sizes = result["s"].as< std::vector<int> >();
    } else if (result.count("levels")) {
      int number_of_levels = result["l"].as<int>();
      level_sizes = std::vector<int>(number_of_levels,12);
    }
	
	int network_optimizer = result["o"].as<int>();

	int epochs = epochs = result["e"].as<int>();

	float lr = result["r"].as<float>();

	int batchSize = result["b"].as<int>();

	/*
     * Input data set handling
     */
    if (!result.count("dir")) {
      std::cout << "Must specify directory with -d/--dir option" << std::endl;
      return 1;
    }
    string mnist_loc = result["dir"].as<string>();
    //std::cout << mnist_loc << std::endl;
    Dataset data;
    data.readTest(mnist_loc.data()); // Placed MNIST in a neighbor directory

    // Parent
    //   |__mnist
    //   |      |_data0
    //   |      |_...
    //   |      |_data9
    //   |__src
    //          |_test.cpp  [You are here]

	if(comm_world.rank()==0)
      cout << data.items.size() << endl; // Show size

    data.shuffle();

    //    Vector v1 = data.items.at(0).data;
    //    Vector gT = data.items.at(0).label;

    Net test_net(data); //v1.size());
    //test_net.addLayer(256, RELU);
    //test_net.addLayer(64, RELU);
    //test_net.addLayer(32, RELU);
    if (level_sizes.size()==2) {
      test_net.addLayer(16, RELU);
      test_net.addLayer(10, SMAX);
    } else {
      for ( auto level_size : level_sizes ) {
	std::cout << "Adding level of size " << level_size << std::endl;
	test_net.addLayer(level_size,RELU);
      }
      test_net.addLayer(10, SMAX);
    }

    auto [trainSplit,testSplit] = data.split(0.95);
	if(comm_world.rank()==0)
      std::cout << "Split " << trainSplit.items.size() << " " << testSplit.items.size() << "\n";
    auto start_time = myclock::now();
 //   std::cout << test_net.accuracy(data) << "\n";

    test_net.trainmpi(trainSplit,testSplit,
	lr, epochs, (opt)network_optimizer, cce,  batchSize, 0.9);
    
	comm_world.barrier();
    auto duration = myclock::now()-start_time;
    auto microsec_duration = std::chrono::duration_cast<std::chrono::microseconds>(duration);
   	 
	if(comm_world.rank()==0){
	  std::cout << "\nFinal Accuracy over all data: " << test_net.accuracy(data) << "\n"
	      << "    attained in " << microsec_duration.count() << "usec" << "\n";
		
      test_net.saveModel("weights.bin");
      test_net.info();
	}
    return 0;
}
#endif

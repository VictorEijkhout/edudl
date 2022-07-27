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

#include <mpl/mpl.hpp>

void Net::trainmpi(Dataset &data, Dataset &testData, float lr, int epochs, opt Optimizer, lossfn lossFuncName, int batchSize, float momentum, float decay) {
	const mpl::communicator &comm_world(mpl::environment::comm_world());	
	lossFunction = lossFunctions.at(lossFuncName);
	int rank = comm_world.rank();
	int size = comm_world.size();
	if(rank==0) {
	std::cout << "Optimizing with ";
	  switch (Optimizer) {
	  case sgd:  std::cout << "Stochastic Gradient Descent\n";  break;
	  case rms:  std::cout << "RMSprop\n"; break;
	  }
	}
	int ssize = batchSize;//data.items.size();
	std::vector<Dataset> batches = data.batch(ssize);
	for (int i = 0; i < batches.size(); i++) {
		batches.at(i).stack(); // Put batch items into one matrix
	}
    
    float loss, acc;
    float lrInit = lr;	
   	
    for (int i_epoch = 0; i_epoch < epochs; i_epoch++) {
      // Iterate through the entire dataset for each epoch
      if(rank==0)
	    std::cout << std::endl << "Epoch " << i_epoch+1 << "/" << epochs;
      lr = lrInit; // Reset the learning rate to undo decay
	  int batchStart = batches.size() * rank / comm_world.size();
	  int batchEnd = batches.size() * (rank + 1) / comm_world.size();
	   
	  for(int idx=batchStart; idx<batchEnd; idx++) {
        calcGrad(batches.at(idx).dataBatch, batches.at(idx).labelBatch);
		for(auto &layer : this->layers) {
		  mpl::contiguous_layout<float> dw_layout(layer.dw.r * layer.dw.c);
		  mpl::contiguous_layout<float> db_layout(layer.db.size());
		  if(rank==0) {
		    Matrix tempdw(layer.dw.r, layer.dw.c, 0);
			Vector tempdb(layer.db.size(), 0);  
			Matrix currdw = layer.dw;
			Vector currdb = layer.db;
			comm_world.reduce(mpl::plus<float>(), 0, tempdw.data(), layer.dw.data(), dw_layout);
			comm_world.reduce(mpl::plus<float>(), 0, tempdb.data(), layer.db.data(), db_layout);
			layer.dw = layer.dw + currdw; // IM the on-rank values get discarded when reducing, or may be a misinterpretation on my side
			layer.db = layer.db + currdb;
			layer.dw = layer.dw / size;
			layer.db = layer.db / size;
		  } else {
		    comm_world.reduce(mpl::plus<float>(), 0, layer.dw.data(), dw_layout);
		    comm_world.reduce(mpl::plus<float>(), 0, layer.db.data(), db_layout);
		  }
	    }
		comm_world.barrier(); //sync before processing the next batch	
        if(rank==0) {
			lr = lr / (1 + decay * idx);	
        	optimize.at(Optimizer)(lr, momentum); 
	    }
	  }
	}
}

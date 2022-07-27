# EduDL

Deep Learning library for educational purposes.

Initialize a new network using net.h. Building a dense network looks something like so:
```C++
Net bob(inputW); // Provide the input size here
bob.addDense(784, RELU); 
bob.addDense(64, RELU);
bob.addDense(10, RELU);
bob.train_vec(trainSet, testSet, learning_rate, epochs, optimizer, batchSize, momentum, learning_decay);
```
# Currently supports 
- [x] Basic linear algebra
- [x] Feedforward ANN
- [x] Stochastic, Mini-batch Training
- [x] Optimizers like Gradient Descent with Momentum, RMSprop
- [x] Nonlinearities like RELU, Sigmoid, SoftMax
- [x] Custom dataset operations like shuffling, splitting, batching, custom reader for MNIST, etc.
- [x] BLIS Support

# neural-network-mockup

## What is this for?
- This a test where people can expirement and throw around ideas and learn about neural networks.
- Check out the projects page: `https://github.com/NonL4331/neural-network-mockup/projects/1`

## How to run
1. Make sure you have rust installed. You can check this by opening a terminal and typing `cargo`. If you do not have rust installed follow the instructions at `https://rustup.rs/`. You may also need to install a fortran compiler.
2. Download the repository and unzip it. If you have git installed `https://git-scm.com/downloads` you can also clone the repository with `git clone https://github.com/NonL4331/neural-network-mockup.git` in a terminal.
3. Navigate to the directory of the repository in a terminal and type `cargo run --example mnist --release`

## What does running the program do?
The first thing that the program does is create a neural network with the specifications specified in `examples/mnst/main.rs`. The program then imports the training and test data from the mnist folder. The data consists of an input which is this case is a flattened 28x28 greyscale image and an expected output which is a number from 0-9 which matches what the number on the image. 

The program takes the 50,000 training images and "learns" from them then attempts to classify 10,000 test images as a number between 0-9. In the terminal you will see output in the form of "epoch x: y/10000 (z%)". The x is the epoch number which is the number of time the train-test process has happened. The y is the number of images classified correctly and z is the percent of images classified correctly.  

Currently the program is set to import the MNIST database to test with `http://yann.lecun.com/exdb/mnist/` which is a database of 60,000 hand written digits as 28x28 greyscale images. 

## I want change the parameters, how do I do that?
### SGD parameters
The paramaters for stochastic gradient descent (SGD) are as follows:

`network.sgd(training_data, Option<test_data>, number_epochs, mini_batch_size, learning_rate)`
| Variable        | Explaination                                                                        |
|-----------------|-------------------------------------------------------------------------------------|
| network         | The neural network itself (note input/output size has to match with data)           |
| training_data   | The data that the network trains/learns with                                        |
| test_data       | The data that the network tests itself with (it doesn't learn with this data)       |
| number_epochs   | The number of epochs to train the network for                                       |
| mini_batch_size | The number of images in each mini batch (think of it like a mini training data set) |
| learning_rate   | How much the network changes with each update                                       |

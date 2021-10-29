use crate::mnist_import::parse_files;
use crate::network::Network;

use crate::network::layer::InitType;
use crate::network::neuron::activation_function::ActivationFunction;

mod network;

mod mnist_import;

fn main() {
    let mut network = Network::from_layers(vec![
        input!(784),
        feedforward!(
            InitType::NormalisedXavier,
            ActivationFunction::Sigmoid,
            784,
            100
        ),
        output!(
            InitType::NormalisedXavier,
            ActivationFunction::Sigmoid,
            100,
            10
        ),
    ]);
    let training_data = parse_files(
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte",
    )
    .unwrap();
    let test_data = parse_files(
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte",
    )
    .unwrap();
    network.sgd(training_data, Some(test_data), 30, 10, 3.0);
}

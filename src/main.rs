mod network;

mod mnist_import;

use crate::mnist_import::parse_files;

use crate::network::{ActivationFunction, CostFunction, InitType, Network};

fn main() {
    let mut network = Network::from_layers(vec![
        input!(784),
        feedforward!(
            ActivationFunction::Sigmoid,
            InitType::NormalisedXavier,
            784,
            100
        ),
        output!(
            ActivationFunction::Sigmoid,
            CostFunction::Quadratic,
            InitType::NormalisedXavier,
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

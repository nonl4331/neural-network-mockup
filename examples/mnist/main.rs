#[macro_use]
extern crate neural_network;

mod mnist_import;

use mnist_import::parse_files;

use neural_network::{ActivationFunction, CostFunction, InitType, Network};

fn main() {
	let mut network = Network::new(vec![
		input!(784),
		feedforward!(ActivationFunction::Sigmoid, InitType::NormalisedXavier, 30),
		output!(
			ActivationFunction::Sigmoid,
			CostFunction::CrossEntropy,
			InitType::NormalisedXavier,
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

	network.sgd(
		training_data,
		Some(test_data),
		30,
		10,
		0.25,
		Some("mnist results"),
	);
}

mod change;
pub mod layer;
mod neuron;
mod utility;

use crate::front_end::graph_results;

pub use neuron::{
	activation_function::ActivationFunction,
	cost_function::{CostFunction, Regularisation},
	initialisation::InitType,
};

pub use utility::Float;
use {
	layer::{
		feedforward::FeedForward, inputlayer::InputLayer, outputlayer::OutputLayer, Layer,
		LayerInfo, LayerInfoTrait, LayerTrait,
	},
	utility::max_index,
};

use rand::prelude::SliceRandom;
use rand::thread_rng;

pub type NetworkData = Vec<(Vec<Float>, Vec<Float>)>;

pub struct Network {
	layers: Vec<Layer>,
}

impl Network {
	fn apply_layer_changes(&mut self, learning_rate: Float, mini_batch_size: usize) {
		for layer in self.layers.iter_mut() {
			layer.update(
				learning_rate,
				mini_batch_size,
				&crate::network::Regularisation::None,
			);
		}
	}

	fn backpropagation(&mut self, input: &Vec<Float>, expected_output: &Vec<Float>) -> Vec<Float> {
		let len = self.layers.len();
		let mut z_values = Vec::new();
		let mut outputs = Vec::new();
		let mut output = input.clone();
		for layer in &mut self.layers {
			layer.forward(output);
			output = layer.last_output();
			outputs.push(output.clone());
			z_values.push(layer.last_z_values());
		}

		let o_len = expected_output.len();

		let (mut errors, mut weights, mut dim) = self.layers[len - 1].backward(
			&outputs[outputs.len() - 2],
			&output,
			(expected_output.clone(), [o_len, 1, 1]),
		);

		for i in 0..(len - 1) {
			let layer_index = len - (i + 2);

			let outputs_index = if i == len - 2 { 0 } else { len - (i + 3) };

			let (terrors, tweights, tweights_dim) =
				self.layers[layer_index].backward(&outputs[outputs_index], &errors, (weights, dim));

			errors = terrors;
			weights = tweights;
			dim = tweights_dim;
		}

		output
	}

	pub fn forward(&mut self, input: Vec<Float>) -> Vec<Float> {
		let mut next_input = input;
		for layer in &mut self.layers {
			layer.forward(next_input);
			next_input = layer.last_output();
		}
		next_input
	}

	fn from_layers(layers: Vec<Layer>) -> Self {
		Network { layers }
	}

	pub fn new(layer_infos: Vec<LayerInfo>) -> Self {
		let mut layers = Vec::new();
		let info_len = layer_infos.len();

		if info_len == 0 {
			panic!("Can't create an empty Network!");
		}

		// handle input layer seperately
		match &layer_infos[0] {
			LayerInfo::InputLayer(info) => {
				layers.push(Layer::InputLayer(InputLayer::new(*info)));
			}
			_ => panic!("Attempting to create network where first layer isn't input!"),
		}

		let mut previous_layer = &layer_infos[0];

		// next_layer might not be needed?
		for (layer, _next_layer) in layer_infos[1..(info_len - 1)]
			.iter()
			.zip(&layer_infos[2..info_len])
		{
			// note this is variable to change as different network types are supported
			match layer {
				LayerInfo::InputLayer(_) => {
					panic!("Attempting to create input layer in middle of network!")
				}
				LayerInfo::FeedForward(info) => layers.push(Layer::FeedForward(FeedForward::new(
					*info,
					previous_layer.flattened_output(),
				))),

				LayerInfo::OutputLayer(_) => {
					panic!("Attempting to create output layer in middle of network!")
				}
			}

			previous_layer = layer;
		}

		// handle output layer seperately
		match &layer_infos[layer_infos.len() - 1] {
			LayerInfo::OutputLayer(info) => layers.push(Layer::OutputLayer(OutputLayer::new(
				*info,
				previous_layer.flattened_output(),
			))),

			_ => panic!("Attempting to create network where first layer isn't input!"),
		}

		Network::from_layers(layers)
	}

	pub fn sgd(
		&mut self,
		mut training_data: NetworkData,
		test_data: Option<NetworkData>,
		epochs: usize,
		mini_batch_size: usize,
		learning_rate: Float,
		graph_output: Option<&str>,
	) {
		let mut results: Vec<(f32, f64)> = Vec::new();
		if test_data.is_some() {
			let mut correct = 0;
			let num = test_data.as_ref().unwrap().len();

			for data in test_data.as_ref().unwrap() {
				let output = self.forward(data.0.clone());
				if max_index(&output) == max_index(&data.1) {
					correct += 1;
				}
			}

			let percent_correct = (correct * 100) as Float / num as Float;
			println!("Epoch 0: {} / {} ({}%)", correct, num, percent_correct);
		}

		let mut max_correct = 0;
		let mut min_correct = test_data.as_ref().unwrap().len();
		for i in 0..epochs {
			training_data.shuffle(&mut thread_rng());

			let mini_batches = training_data.chunks(mini_batch_size);

			for mini_batch in mini_batches {
				for data in mini_batch {
					self.backpropagation(&data.0, &data.1);
				}
				self.apply_layer_changes(learning_rate, mini_batch_size);
			}

			if test_data.is_some() {
				let mut correct = 0;
				let num = test_data.as_ref().unwrap().len();

				for data in test_data.as_ref().unwrap() {
					let output = self.forward(data.0.clone());
					if max_index(&output) == max_index(&data.1) {
						correct += 1;
					}
				}
				if correct > max_correct {
					max_correct = correct;
				}

				if correct < min_correct {
					min_correct = correct;
				}

				let percent_correct = (correct * 100) as Float / num as Float;
				println!(
					"Epoch {}: {} / {} ({}%)",
					i + 1,
					correct,
					num,
					percent_correct
				);
				results.push(((i + 1) as Float, percent_correct as f64));
			} else {
				println!("Epoch {} complete.", i + 1);
			}
		}
		if test_data.is_some() {
			let num = test_data.as_ref().unwrap().len();
			println!(
				"Highest accuracy: {} / {} ({}%)",
				max_correct,
				num,
				(max_correct * 100) as Float / num as Float
			);
			match graph_output {
				Some(name) => {
					graph_results(
						name,
						epochs,
						results,
						(min_correct * 100) as f64 / num as f64,
						(max_correct * 100) as f64 / num as f64,
					);
				}
				_ => {}
			}
		}
	}
}

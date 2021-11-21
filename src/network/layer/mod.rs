pub mod feedforward;
pub mod inputlayer;
pub mod outputlayer;

use crate::network::{Float, Regularisation};

use {
	feedforward::{FeedForward, FeedForwardInfo},
	inputlayer::{InputLayer, InputLayerInfo},
	outputlayer::{OutputLayer, OutputLayerInfo},
};

pub enum LayerInfo {
	FeedForward(FeedForwardInfo),
	InputLayer(InputLayerInfo),
	OutputLayer(OutputLayerInfo),
}

pub enum Layer {
	FeedForward(FeedForward),
	InputLayer(InputLayer),
	OutputLayer(OutputLayer),
}

impl LayerInfoTrait for LayerInfo {
	fn output(&self) -> [usize; 3] {
		match self {
			LayerInfo::FeedForward(info) => [info.length, 1, 1],
			LayerInfo::InputLayer(info) => info.sizes,
			LayerInfo::OutputLayer(info) => [info.length, 1, 1],
		}
	}
}

impl LayerTrait for Layer {
	fn backward(
		&mut self,
		a: &[Float],
		error_input: &[Float],
		weights: (Vec<Float>, [usize; 3]),
	) -> (Vec<Float>, Vec<Float>, [usize; 3]) {
		match self {
			Layer::InputLayer(layer) => (*layer).backward(a, error_input, weights),
			Layer::FeedForward(layer) => (*layer).backward(a, error_input, weights),
			Layer::OutputLayer(layer) => (*layer).backward(a, error_input, weights),
		}
	}

	fn forward(&mut self, input: Vec<Float>) {
		match self {
			Layer::FeedForward(layer) => layer.forward(input),
			Layer::InputLayer(layer) => layer.forward(input),
			Layer::OutputLayer(layer) => layer.forward(input),
		}
	}

	fn last_output(&self) -> Vec<Float> {
		match self {
			Layer::FeedForward(layer) => (*layer).last_output(),
			Layer::InputLayer(layer) => (*layer).last_output(),
			Layer::OutputLayer(layer) => (*layer).last_output(),
		}
	}

	fn last_z_values(&self) -> Vec<Float> {
		match self {
			Layer::FeedForward(layer) => (*layer).last_z_values(),
			Layer::InputLayer(layer) => (*layer).last_z_values(),
			Layer::OutputLayer(layer) => (*layer).last_z_values(),
		}
	}

	fn update(
		&mut self,
		learning_rate: Float,
		mini_batch_size: usize,
		regularisation: &Regularisation,
	) {
		match self {
			Layer::FeedForward(layer) => {
				(*layer).update(learning_rate, mini_batch_size, regularisation)
			}
			Layer::InputLayer(layer) => {
				(*layer).update(learning_rate, mini_batch_size, regularisation)
			}
			Layer::OutputLayer(layer) => {
				(*layer).update(learning_rate, mini_batch_size, regularisation)
			}
		}
	}

	fn update_change(&mut self, errors: &[Float], a: &[Float]) {
		match self {
			Layer::FeedForward(layer) => (*layer).update_change(errors, a),
			Layer::InputLayer(layer) => (*layer).update_change(errors, a),
			Layer::OutputLayer(layer) => (*layer).update_change(errors, a),
		}
	}
}

pub trait LayerInfoTrait {
	fn flattened_output(&self) -> usize {
		let output = self.output();
		output[0] * output[1] * output[2]
	}

	fn output(&self) -> [usize; 3];
}

pub trait LayerTrait {
	fn backward(
		&mut self,
		a: &[Float],
		error_input: &[Float],
		weights: (Vec<Float>, [usize; 3]),
		// errors, weights, dimensions_weights
	) -> (Vec<Float>, Vec<Float>, [usize; 3]);
	fn forward(&mut self, input: Vec<Float>);
	fn last_output(&self) -> Vec<Float>;
	fn last_z_values(&self) -> Vec<Float>;
	fn update(
		&mut self,
		learning_rate: Float,
		mini_batch_size: usize,
		regularisation: &Regularisation,
	);
	fn update_change(&mut self, errors: &[Float], a: &[Float]);
}

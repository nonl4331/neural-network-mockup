use crate::network::{Float, Regularisation};

use super::LayerTrait;

pub struct InputLayer {
	length: usize,
	output: Vec<Float>,
}

impl LayerTrait for InputLayer {
	fn backward(
		&mut self,
		_: &[Float],
		error_input: &[Float],
		weights: (Vec<Float>, [usize; 3]),
	) -> (Vec<Float>, Vec<Float>, [usize; 3]) {
		(error_input.to_vec(), weights.0, weights.1)
	}

	fn forward(&mut self, input: Vec<Float>) {
		assert_eq!(self.length, input.len());

		self.output = input;
	}

	fn last_output(&self) -> Vec<Float> {
		self.output.clone()
	}

	fn last_z_values(&self) -> Vec<Float> {
		self.output.clone()
	}

	fn update(&mut self, _: Float, _: usize, _: &Regularisation) {}

	fn update_change(&mut self, _: &[Float], _: &[Float]) {}
}

impl InputLayer {
	pub fn new(length: usize) -> Self {
		InputLayer {
			length,
			output: Vec::new(),
		}
	}
}

#[macro_export]
macro_rules! input {
	($length:expr) => {
		neural_network::layer::Layer::InputLayer(
			neural_network::layer::inputlayer::InputLayer::new($length),
		)
	};
}

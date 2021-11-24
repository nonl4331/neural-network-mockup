use crate::network::{Float, Regularisation};

use super::{LayerInfoTrait, LayerTrait};

#[derive(Copy, Clone)]
pub struct InputLayerInfo {
	pub sizes: [usize; 3],
}

pub struct InputLayer {
	info: InputLayerInfo,
	output: Vec<Float>,
}

impl InputLayerInfo {
	pub fn new(sizes: [usize; 3]) -> Self {
		InputLayerInfo { sizes }
	}
}

impl LayerInfoTrait for InputLayerInfo {
	fn output(&self) -> [usize; 3] {
		self.sizes
	}
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
		assert_eq!(self.info.flattened_output(), input.len());

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
	pub fn new(info: InputLayerInfo) -> Self {
		InputLayer {
			info,
			output: Vec::new(),
		}
	}
}

#[macro_export]
macro_rules! input {
	($length:expr) => {
		neural_network::layer::LayerInfo::InputLayer(
			neural_network::layer::inputlayer::InputLayerInfo::new([$length, 1, 1]),
		)
	};
}

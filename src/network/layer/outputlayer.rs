use crate::network::change::OutputLayerChange;
use crate::network::utility::{
	matrix_multiply_sum, outer_product_add, plus_equals_matrix_multiplied, scale_elements,
};
use crate::network::{ActivationFunction, CostFunction, Float, InitType, Regularisation};

extern crate openblas_src;

use super::{LayerInfoTrait, LayerTrait};

#[derive(Copy, Clone)]
pub struct OutputLayerInfo {
	pub activation_function: ActivationFunction,
	pub cost_function: CostFunction,
	pub init_type: InitType,
	pub length: usize,
}

pub struct OutputLayer {
	biases: Vec<Float>,
	change: Option<OutputLayerChange>,
	info: OutputLayerInfo,
	output: Vec<Float>,
	weights: Vec<Float>,
	weight_dimensions: [usize; 2],
	z_values: Vec<Float>,
}

impl OutputLayerInfo {
	pub fn new(
		activation_function: ActivationFunction,
		cost_function: CostFunction,
		init_type: InitType,
		length: usize,
	) -> Self {
		OutputLayerInfo {
			activation_function,
			cost_function,
			init_type,
			length,
		}
	}
}

impl LayerInfoTrait for OutputLayerInfo {
	fn output(&self) -> [usize; 3] {
		[self.length, 1, 1]
	}
}

impl LayerTrait for OutputLayer {
	fn backward(
		&mut self,
		a: &[Float],
		output: &[Float],
		expected_output: (Vec<Float>, [usize; 3]),
	) -> (Vec<Float>, Vec<Float>, [usize; 3]) {
		let expected_output = &expected_output.0;

		let errors: Vec<Float> = output
			.iter()
			.zip(expected_output.iter())
			.zip(self.z_values.iter())
			.map(|((output, expected_output), z)| {
				self.info.cost_function.c_dz(
					&self.info.activation_function,
					output,
					expected_output,
					z,
				)
			})
			.collect();

		self.update_change(&errors, a);

		(
			errors,
			self.weights.clone(),
			[self.weight_dimensions[0], self.weight_dimensions[1], 1],
		)
	}

	fn forward(&mut self, input: Vec<Float>) {
		assert_eq!(self.weight_dimensions[1], input.len());
		self.z_values = self.biases.clone();

		// change to matrix vector operation?
		matrix_multiply_sum(
			&self.weights,
			&input,
			self.weight_dimensions,
			&mut self.z_values,
		);

		// is this optimal??
		// perhaps have activation_function(z_values: &[]) -> Vec<>
		let output = self
			.z_values
			.iter()
			.map(|&z| self.info.activation_function.evaluate(z))
			.collect();

		self.output = output;
	}

	fn last_output(&self) -> Vec<Float> {
		self.output.clone()
	}

	fn last_z_values(&self) -> Vec<Float> {
		self.z_values.clone()
	}

	fn update(
		&mut self,
		learning_rate: Float,
		mini_batch_size: usize,
		regularisation: &Regularisation,
	) {
		match &self.change {
			Some(change) => match regularisation {
				Regularisation::L1(lambda) => {
					let multiplier = -1.0 / mini_batch_size as Float;

					scale_elements(&mut self.weights, 1.0 + learning_rate * lambda * multiplier);

					plus_equals_matrix_multiplied(&mut self.weights, multiplier, &change.weights);
				}
				Regularisation::L2(lambda) => {
					let multiplier = -learning_rate / mini_batch_size as Float;

					// is there a more optimal way to do this?
					for (weight, change) in self.weights.iter_mut().zip(change.weights.iter()) {
						*weight += multiplier * (change + lambda * weight.signum());
					}
				}
				Regularisation::None => {
					let multiplier = -learning_rate / mini_batch_size as Float;

					plus_equals_matrix_multiplied(&mut self.weights, multiplier, &change.weights);

					plus_equals_matrix_multiplied(&mut self.biases, multiplier, &change.biases);
				}
			},
			None => {}
		}
		self.change = Some(OutputLayer::empty_layer_change(&self.weight_dimensions));
	}

	fn update_change(&mut self, errors: &[Float], a: &[Float]) {
		let change = self.change.as_mut().unwrap();
		let biases = &mut change.biases;
		let weights = &mut change.weights;

		for (bias, error) in biases.iter_mut().zip(errors) {
			*bias += error;
		}

		outer_product_add(&errors, &a, weights);
	}
}

impl OutputLayer {
	pub fn new(info: OutputLayerInfo, input_size: usize) -> Self {
		let biases = vec![0.0; info.length];
		let mut weights = Vec::new();
		for _ in 0..info.length * input_size {
			weights.push(info.init_type.generate_weight(input_size, info.length));
		}

		let weight_dimensions = [info.length, input_size];

		OutputLayer {
			biases,
			change: Some(OutputLayer::empty_layer_change(&weight_dimensions)),
			info,
			output: Vec::new(),
			weights,
			weight_dimensions,
			z_values: Vec::new(),
		}
	}
	fn empty_layer_change(weight_dim: &[usize; 2]) -> OutputLayerChange {
		OutputLayerChange::new(weight_dim)
	}
}

#[macro_export]
macro_rules! output {
	($activation_function:expr, $cost_function:expr, $init_type:expr, $length:expr) => {
		neural_network::layer::LayerInfo::OutputLayer(
			neural_network::layer::outputlayer::OutputLayerInfo::new(
				$activation_function,
				$cost_function,
				$init_type,
				$length,
			),
		)
	};
}

use crate::network::{ActivationFunction, Float, InitType, Regularisation};

use crate::network::change::FeedForwardChange;

use crate::network::utility::{
	hadamard_product, matrix_vec_multiply_add, outer_product_add, plus_equals_matrix_multiplied,
	scale_elements, transpose_matrix_multiply_vec,
};

extern crate openblas_src;

use super::{LayerInfoTrait, LayerTrait};

#[derive(Copy, Clone)]
pub struct FeedForwardInfo {
	activation_function: ActivationFunction,
	init_type: InitType,
	pub length: usize,
}

pub struct FeedForward {
	biases: Vec<Float>,
	change: Option<FeedForwardChange>,
	info: FeedForwardInfo,
	output: Vec<Float>,
	weights: Vec<Float>,
	weight_dimensions: [usize; 2],
	z_values: Vec<Float>,
}

impl FeedForwardInfo {
	pub fn new(
		activation_function: ActivationFunction,
		init_type: InitType,
		length: usize,
	) -> Self {
		FeedForwardInfo {
			activation_function,
			init_type,
			length,
		}
	}
}

impl LayerInfoTrait for FeedForwardInfo {
	fn output(&self) -> [usize; 3] {
		[self.length, 1, 1]
	}
}

impl LayerTrait for FeedForward {
	fn backward(
		&mut self,
		a: &[Float],
		error_input: &[Float],
		weights: (Vec<Float>, [usize; 3]),
	) -> (Vec<Float>, Vec<Float>, [usize; 3]) {
		let mut c_da = Vec::new();

		assert_eq!(weights.1[2], 1);

		let weights = (weights.0, [weights.1[0], weights.1[1]]);

		transpose_matrix_multiply_vec(&weights.0, error_input, weights.1, &mut c_da);

		let a_dz: Vec<Float> = self
			.z_values
			.iter()
			.map(|&z| self.info.activation_function.derivative(z))
			.collect();

		let errors = hadamard_product(&c_da, &a_dz);

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
		matrix_vec_multiply_add(
			&self.weights,
			&input,
			&mut self.z_values,
			&self.weight_dimensions,
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
		self.change = Some(FeedForward::empty_layer_change(&self.weight_dimensions));
	}

	fn update_change(&mut self, errors: &[Float], a: &[Float]) {
		assert_eq!(self.weight_dimensions[0], errors.len());
		// On entry to SGER   parameter number  9 had an illegal value
		// prob need to write unit tests for backwards
		let change = self.change.as_mut().unwrap();
		let biases = &mut change.biases;
		let weights = &mut change.weights;

		for (bias, error) in biases.iter_mut().zip(errors) {
			*bias += error;
		}

		outer_product_add(&errors, &a, weights);
	}
}

impl FeedForward {
	pub fn new(info: FeedForwardInfo, input_size: usize) -> Self {
		let biases = vec![0.0; info.length];
		let mut weights = Vec::new();
		for _ in 0..info.length * input_size {
			weights.push(info.init_type.generate_weight(input_size, info.length));
		}

		let weight_dimensions = [info.length, input_size];

		FeedForward {
			biases,
			change: Some(FeedForward::empty_layer_change(&weight_dimensions)),
			info,
			output: Vec::new(),
			weights,
			weight_dimensions,
			z_values: Vec::new(),
		}
	}

	fn empty_layer_change(weight_dim: &[usize; 2]) -> FeedForwardChange {
		FeedForwardChange::new(weight_dim)
	}
}

#[macro_export]
macro_rules! feedforward {
	($activation_function:expr, $init_type:expr, $length:expr) => {
		neural_network::layer::LayerInfo::FeedForward(
			neural_network::layer::feedforward::FeedForwardInfo::new(
				$activation_function,
				$init_type,
				$length,
			),
		)
	};
}

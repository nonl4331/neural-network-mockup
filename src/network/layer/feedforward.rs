use crate::network::{ActivationFunction, Float, InitType, Regularisation};

use crate::network::change::FeedForwardChange;

use crate::network::utility::{
	hadamard_product, matrix_vec_multiply_add, outer_product_add, plus_equals_matrix_multiplied,
	scale_elements, transpose_matrix_multiply_vec,
};

extern crate openblas_src;

use super::{LayerInfoTrait, LayerTrait};

pub struct FeedForwardData {
	biases: Vec<Float>,
	weights: Vec<Float>,
	weight_dimensions: [usize; 2],
}

#[derive(Copy, Clone)]
pub struct FeedForwardInfo {
	activation_function: ActivationFunction,
	init_type: InitType,
	pub length: usize,
}

pub struct FeedForwardOutput {
	after_activation: Vec<Float>,
	before_activation: Vec<Float>,
}

pub struct FeedForward {
	change: Option<FeedForwardChange>,
	data: FeedForwardData,
	info: FeedForwardInfo,
	outputs: FeedForwardOutput,
}

impl FeedForwardData {
	pub fn new(init_type: InitType, weight_dimensions: [usize; 2]) -> Self {
		let biases = vec![0.0; weight_dimensions[0]];
		let mut weights = Vec::new();
		for _ in 0..(weight_dimensions[0] * weight_dimensions[1]) {
			weights.push(init_type.generate_weight(weight_dimensions[1], weight_dimensions[0]));
		}

		FeedForwardData {
			biases,
			weights,
			weight_dimensions,
		}
	}
}

impl FeedForwardOutput {
	pub fn new() -> Self {
		FeedForwardOutput {
			after_activation: Vec::new(),
			before_activation: Vec::new(),
		}
	}
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
			.outputs
			.before_activation
			.iter()
			.map(|&z| self.info.activation_function.derivative(z))
			.collect();

		let errors = hadamard_product(&c_da, &a_dz);

		self.update_change(&errors, a);

		(
			errors,
			self.data.weights.clone(),
			[
				self.data.weight_dimensions[0],
				self.data.weight_dimensions[1],
				1,
			],
		)
	}

	fn forward(&mut self, input: Vec<Float>) {
		assert_eq!(self.data.weight_dimensions[1], input.len());

		self.outputs.before_activation = self.data.biases.clone();
		matrix_vec_multiply_add(
			&self.data.weights,
			&input,
			&mut self.outputs.before_activation,
			&self.data.weight_dimensions,
		);

		// is this optimal??
		// perhaps have activation_function(z_values: &[]) -> Vec<>
		let output = self
			.outputs
			.before_activation
			.iter()
			.map(|&z| self.info.activation_function.evaluate(z))
			.collect();

		self.outputs.after_activation = output;
	}

	fn last_output(&self) -> Vec<Float> {
		self.outputs.after_activation.clone()
	}

	fn last_z_values(&self) -> Vec<Float> {
		self.outputs.before_activation.clone()
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

					scale_elements(
						&mut self.data.weights,
						1.0 + learning_rate * lambda * multiplier,
					);

					plus_equals_matrix_multiplied(
						&mut self.data.weights,
						multiplier,
						&change.weights,
					);
				}
				Regularisation::L2(lambda) => {
					let multiplier = -learning_rate / mini_batch_size as Float;

					// is there a more optimal way to do this?
					for (weight, change) in self.data.weights.iter_mut().zip(change.weights.iter())
					{
						*weight += multiplier * (change + lambda * weight.signum());
					}
				}
				Regularisation::None => {
					let multiplier = -learning_rate / mini_batch_size as Float;

					plus_equals_matrix_multiplied(
						&mut self.data.weights,
						multiplier,
						&change.weights,
					);

					plus_equals_matrix_multiplied(
						&mut self.data.biases,
						multiplier,
						&change.biases,
					);
				}
			},
			None => {}
		}
		self.change = Some(FeedForward::empty_layer_change(
			&self.data.weight_dimensions,
		));
	}

	fn update_change(&mut self, errors: &[Float], a: &[Float]) {
		assert_eq!(self.data.weight_dimensions[0], errors.len());
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
		let weight_dimensions = [info.length, input_size];

		let data = FeedForwardData::new(info.init_type, weight_dimensions);

		FeedForward {
			change: Some(FeedForward::empty_layer_change(&weight_dimensions)),
			data,
			info,
			outputs: FeedForwardOutput::new(),
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

use crate::network::change::OutputLayerChange;
use crate::network::utility::{
	matrix_multiply_sum, outer_product_add, plus_equals_matrix_multiplied, scale_elements,
};
use crate::network::{ActivationFunction, CostFunction, Float, InitType, Regularisation};

extern crate openblas_src;

use super::{LayerInfoTrait, LayerTrait};

pub struct OutputLayerData {
	biases: Vec<Float>,
	weights: Vec<Float>,
	weight_dimensions: [usize; 2],
}

#[derive(Copy, Clone)]
pub struct OutputLayerInfo {
	pub activation_function: ActivationFunction,
	pub cost_function: CostFunction,
	pub init_type: InitType,
	pub length: usize,
}

pub struct OutputLayerOutput {
	after_activation: Vec<Float>,
	before_activation: Vec<Float>,
}

pub struct OutputLayer {
	change: Option<OutputLayerChange>,
	data: OutputLayerData,
	info: OutputLayerInfo,
	outputs: OutputLayerOutput,
}

impl OutputLayerData {
	pub fn new(init_type: InitType, weight_dimensions: [usize; 2]) -> Self {
		let biases = vec![0.0; weight_dimensions[0]];
		let mut weights = Vec::new();
		for _ in 0..(weight_dimensions[0] * weight_dimensions[1]) {
			weights.push(init_type.generate_weight(weight_dimensions[1], weight_dimensions[0]));
		}

		OutputLayerData {
			biases,
			weights,
			weight_dimensions,
		}
	}
}

impl OutputLayerOutput {
	pub fn new() -> Self {
		OutputLayerOutput {
			after_activation: Vec::new(),
			before_activation: Vec::new(),
		}
	}
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
			.zip(self.outputs.before_activation.iter())
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

		// change to matrix vector operation?
		matrix_multiply_sum(
			&self.data.weights,
			&input,
			self.data.weight_dimensions,
			&mut self.outputs.before_activation,
		);

		// is this optimal??
		// perhaps have activation_function(outputs.before_activation: &[]) -> Vec<>
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
		self.change = Some(OutputLayer::empty_layer_change(
			&self.data.weight_dimensions,
		));
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
		let weight_dimensions = [info.length, input_size];

		let data = OutputLayerData::new(info.init_type, weight_dimensions);

		OutputLayer {
			change: Some(OutputLayer::empty_layer_change(&weight_dimensions)),
			data,
			info,
			outputs: OutputLayerOutput::new(),
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

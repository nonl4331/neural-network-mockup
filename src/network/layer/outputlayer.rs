use blas::sgemm;

use crate::network::{ActivationFunction, CostFunction, Float, InitType, Regularisation};

use crate::network::change::OutputLayerChange;

use crate::network::utility::{outer_product_add, plus_equals_matrix_multiplied, scale_elements};

extern crate openblas_src;

use super::LayerTrait;

pub struct OutputLayer {
    activation_function: ActivationFunction,
    biases: Vec<Float>,
    change: Option<OutputLayerChange>,
    cost_function: CostFunction,
    output: Vec<Float>,
    weights: Vec<Float>,
    weight_dimensions: [usize; 2],
    z_values: Vec<Float>,
}

impl LayerTrait for OutputLayer {
    fn backward(
        &mut self,
        a: &[Float],
        output: &[Float],
        expected_output: (Vec<Float>, [usize; 2]),
    ) -> (Vec<Float>, Vec<Float>, [usize; 2]) {
        let expected_output = &expected_output.0;

        let errors: Vec<Float> = output
            .iter()
            .zip(expected_output.iter())
            .zip(self.z_values.iter())
            .map(|((output, expected_output), z)| {
                self.cost_function
                    .c_dz(&self.activation_function, output, expected_output, z)
            })
            .collect();

        self.update_change(&errors, a);

        (errors, self.weights.clone(), self.weight_dimensions)
    }

    fn forward(&mut self, input: Vec<Float>) {
        assert_eq!(self.weight_dimensions[1], input.len());
        self.z_values = self.biases.clone();
        unsafe {
            sgemm(
                b'N',
                b'N',
                self.weight_dimensions[0] as i32,
                1,
                self.weight_dimensions[1] as i32,
                1.0,
                &self.weights,
                self.weight_dimensions[0] as i32,
                &input,
                self.weight_dimensions[1] as i32,
                1.0,
                &mut self.z_values,
                self.weight_dimensions[1] as i32,
            );
        }

        // is this optimal??
        // perhaps have activation_function(z_values: &[]) -> Vec<>
        let output = self
            .z_values
            .iter()
            .map(|&z| self.activation_function.evaluate(z))
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
    pub fn new(
        activation_function: ActivationFunction,
        cost_function: CostFunction,
        init_type: InitType,
        input_size: usize,
        length: usize,
    ) -> Self {
        let biases = vec![0.0; length];
        let mut weights = Vec::new();
        for _ in 0..length * input_size {
            weights.push(init_type.generate_weight(input_size, length));
        }

        let weight_dimensions = [length, input_size];

        OutputLayer {
            activation_function,
            biases,
            change: Some(OutputLayer::empty_layer_change(&weight_dimensions)),
            cost_function,
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
    ($activation_function:expr, $cost_function:expr, $init_type:expr, $input_size:expr, $length:expr) => {
        neural_network::layer::Layer::OutputLayer(
            neural_network::layer::outputlayer::OutputLayer::new(
                $activation_function,
                $cost_function,
                $init_type,
                $input_size,
                $length,
            ),
        )
    };
}

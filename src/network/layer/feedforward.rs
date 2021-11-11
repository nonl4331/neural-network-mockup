use crate::network::{ActivationFunction, Float, InitType, Regularisation};

use crate::network::change::FeedForwardChange;

use crate::network::utility::{
    hadamard_product, matrix_vec_multiply_add, outer_product_add, plus_equals_matrix_multiplied,
    transpose_matrix_multiply_vec,
};

extern crate openblas_src;

use super::LayerTrait;

pub struct FeedForward {
    activation_function: ActivationFunction,
    biases: Vec<Float>,
    change: Option<FeedForwardChange>,
    output: Vec<Float>,
    weights: Vec<Float>,
    weight_dimensions: [usize; 2],
    z_values: Vec<Float>,
}

impl LayerTrait for FeedForward {
    fn backward(
        &mut self,
        a: &[Float],
        error_input: &[Float],
        weights: (Vec<Float>, [usize; 2]),
    ) -> (Vec<Float>, Vec<Float>, [usize; 2]) {
        let mut c_da = Vec::new();

        transpose_matrix_multiply_vec(&weights.0, error_input, weights.1, &mut c_da);

        let a_dz: Vec<Float> = self
            .z_values
            .iter()
            .map(|&z| self.activation_function.derivative(z))
            .collect();

        let errors = hadamard_product(&c_da, &a_dz);

        self.update_change(&errors, a);

        (errors, self.weights.clone(), self.weight_dimensions)
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
                Regularisation::L1(_lambda) => {
                    todo!()
                    //let multiplier = -1.0 / mini_batch_size as Float;
                    //self.weights *= 1.0 + learning_rate * lambda * multiplier;
                    //self.weights += multiplier * change;
                }
                Regularisation::L2(_lambda) => {
                    todo!()
                    //let multiplier = -learning_rate / mini_batch_size as Float;
                    //self.weights += multiplier * (change + lambda * weight.signum());
                }
                Regularisation::None => {
                    let multiplier = -learning_rate / mini_batch_size as Float;

                    //self.weights += multiplier * self.change.unwrap().weights;
                    plus_equals_matrix_multiplied(&mut self.weights, multiplier, &change.weights);

                    //self.bias += multiplier * self.change.unwrap().bias;
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
    pub fn new(
        activation_function: ActivationFunction,
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

        FeedForward {
            activation_function,
            biases,
            change: Some(FeedForward::empty_layer_change(&weight_dimensions)),
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
    ($activation_function:expr, $init_type:expr, $input_size:expr, $length:expr) => {
        neural_network::layer::Layer::FeedForward(
            neural_network::layer::feedforward::FeedForward::new(
                $activation_function,
                $init_type,
                $input_size,
                $length,
            ),
        )
    };
}

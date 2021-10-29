use crate::network::{ActivationFunction, Float, InitType, Neuron, Regularisation};

use crate::network::change::{FeedForwardChange, LayerChange};

use crate::network::utility::{get_weights_vec, hadamard_product, transpose};

use super::LayerTrait;

pub struct FeedForward {
    activation_function: ActivationFunction,
    neurons: Vec<Neuron>,
    output: Vec<Float>,
    z_values: Vec<Float>,
}

impl LayerTrait for FeedForward {
    fn backward(
        &mut self,
        a: &Vec<Float>,
        layer_change: &mut LayerChange,
        error_input: &Vec<Float>,
        weights: Vec<Vec<Float>>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let weights = transpose(weights);

        let c_da: Vec<Float> = weights
            .iter()
            .map(|weights| hadamard_product(weights, &error_input).iter().sum())
            .collect();

        let a_dz: Vec<Float> = self
            .z_values
            .iter()
            .map(|&z| self.activation_function.derivative(z))
            .collect();

        let errors = hadamard_product(&c_da, &a_dz);

        layer_change.update(&errors, a);

        (errors, get_weights_vec(&self.neurons))
    }

    fn empty_layer_change(&self) -> LayerChange {
        LayerChange::FeedForwardChange(FeedForwardChange::new(
            self.neurons.len(),
            self.neurons[0].weights.len(),
        ))
    }

    fn forward(&mut self, input: Vec<Float>) {
        assert_eq!(self.neurons[0].weights.len(), input.len());
        let mut z_values = Vec::new();

        for neuron in &self.neurons {
            let z = neuron
                .weights
                .iter()
                .zip(input.iter())
                .fold(0.0, |acc, (input, weight)| acc + input * weight)
                + neuron.bias;

            z_values.push(z);
        }

        let output = z_values
            .iter()
            .map(|&z| self.activation_function.evaluate(z))
            .collect();

        self.z_values = z_values;
        self.output = output;
    }

    fn last_output(&self) -> Vec<Float> {
        self.output.clone()
    }

    fn last_z_values(&self) -> Vec<Float> {
        self.z_values.clone()
    }

    fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    fn update(
        &mut self,
        changes: &LayerChange,
        learning_rate: Float,
        mini_batch_size: usize,
        regularisation: &Regularisation,
    ) {
        for (neuron, neuron_change) in self.neurons.iter_mut().zip(changes.get_neurons()) {
            neuron.update(
                learning_rate,
                neuron_change,
                mini_batch_size,
                regularisation,
            );
        }
    }
}

impl FeedForward {
    pub fn new(
        activation_function: ActivationFunction,
        init_type: InitType,
        input_size: usize,
        length: usize,
    ) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..length {
            let mut weights = Vec::new();
            for _ in 0..input_size {
                weights.push(init_type.generate_weight(input_size, length));
            }
            neurons.push(Neuron::new(weights, 0.0));
        }
        FeedForward {
            activation_function,
            neurons,
            output: Vec::new(),
            z_values: Vec::new(),
        }
    }
}

#[macro_export]
macro_rules! feedforward {
    ($activation_function:expr, $init_type:expr, $input_size:expr, $length:expr) => {
        network::layer::Layer::FeedForward(network::layer::feedforward::FeedForward::new(
            $activation_function,
            $init_type,
            $input_size,
            $length,
        ))
    };
}

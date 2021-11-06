use crate::network::{ActivationFunction, CostFunction, Float, InitType, Neuron, Regularisation};

use crate::network::change::{LayerChange, OutputLayerChange};

use crate::network::utility::get_weights_vec;

use super::LayerTrait;

pub struct OutputLayer {
    activation_function: ActivationFunction,
    cost_function: CostFunction,
    neurons: Vec<Neuron>,
    z_values: Vec<Float>,
    output: Vec<Float>,
}

impl LayerTrait for OutputLayer {
    fn backward(
        &mut self,
        a: &[Float],
        layer_change: &mut LayerChange,
        output: &[Float],
        expected_output: Vec<Vec<Float>>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        assert_eq!(expected_output.len(), 1);
        let expected_output = &expected_output[0];

        let errors: Vec<Float> = output
            .iter()
            .zip(expected_output.iter())
            .zip(self.z_values.iter())
            .map(|((output, expected_output), z)| {
                self.cost_function
                    .c_dz(&self.activation_function, output, expected_output, z)
            })
            .collect();

        layer_change.update(&errors, a);

        (errors, get_weights_vec(&self.neurons))
    }

    fn empty_layer_change(&self) -> LayerChange {
        LayerChange::OutputLayerChange(OutputLayerChange::new(
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

impl OutputLayer {
    pub fn new(
        activation_function: ActivationFunction,
        cost_function: CostFunction,
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
        OutputLayer {
            activation_function,
            cost_function,
            neurons,
            output: Vec::new(),
            z_values: Vec::new(),
        }
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

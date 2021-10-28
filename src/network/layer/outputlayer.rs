use crate::network::change::LayerChange;
use crate::network::change::OutputLayerChange;
use crate::network::{
    layer::{ActivationFunction, InitType, LayerTrait, Neuron},
    utility::{
        d_quadratic_cost, d_sigmoid, get_weights_vec, hadamard_product, sigmoid,
        update_change_neurons, xavier_init, Float,
    },
};

pub struct OutputLayer {
    activation_function: ActivationFunction,
    neurons: Vec<Neuron>,
    z_values: Vec<Float>,
    output: Vec<Float>,
}

impl LayerTrait for OutputLayer {
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
        let output = match self.activation_function {
            ActivationFunction::Sigmoid => z_values.iter().map(|&z| sigmoid(z)).collect(),
            ActivationFunction::Softmax => {
                let sum: Float = z_values.iter().map(|&z| z.exp()).sum();

                z_values.iter().map(|&z| z.exp() / sum).collect()
            }
        };
        self.z_values = z_values;
        self.output = output;
    }
    fn backward(
        &mut self,
        a: &Vec<Float>,
        layer_change: &mut LayerChange,
        output: &Vec<Float>,
        expected_output: Vec<Vec<Float>>,
        eta: Float,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        assert_eq!(expected_output.len(), 1);
        let expected_output = &expected_output[0];
        let c_da: Vec<Float> = output
            .iter()
            .zip(expected_output.iter())
            .map(|(&output, &expected_output)| d_quadratic_cost(output, expected_output))
            .collect();

        let a_dz: Vec<Float> = match self.activation_function {
            ActivationFunction::Sigmoid => self.z_values.iter().map(|&z| d_sigmoid(z)).collect(),
            ActivationFunction::Softmax => {
                panic!("a_dz not implemented for softmax yet!");
            }
        };

        let errors = hadamard_product(&c_da, &a_dz);

        update_change_neurons(layer_change.get_neurons_mut(), &errors, &a, eta);

        (errors, get_weights_vec(&self.neurons))
    }
    fn get_len(&self) -> usize {
        self.neurons.len()
    }
    fn get_output(&self) -> Vec<Float> {
        self.output.clone()
    }
    fn get_z_values(&self) -> Vec<Float> {
        self.z_values.clone()
    }
    fn get_layer_change(&self) -> LayerChange {
        LayerChange::OutputLayerChange(OutputLayerChange::new(
            self.neurons.len(),
            self.neurons[0].weights.len(),
        ))
    }
    fn update(&mut self, changes: &LayerChange, mini_batch_size: usize) {
        for (neuron, neuron_change) in self.neurons.iter_mut().zip(changes.get_neurons()) {
            neuron.update(neuron_change, mini_batch_size);
        }
    }
}

impl OutputLayer {
    pub fn new(
        init_type: InitType,
        activation_function: ActivationFunction,
        input_size: usize,
        length: usize,
    ) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..length {
            let mut weights = Vec::new();
            for _ in 0..input_size {
                match init_type {
                    InitType::Xavier => {
                        weights.push(xavier_init(input_size, 1));
                    }
                }
            }
            neurons.push(Neuron::new(weights, 0.0));
        }
        OutputLayer {
            neurons,
            activation_function,
            z_values: Vec::new(),
            output: Vec::new(),
        }
    }
}

pub mod feedforward;
pub mod inputlayer;
pub mod outputlayer;

use crate::network::change::LayerChange;
use crate::network::change::NeuronChange;
use crate::network::utility::{update_bias, update_weights, Float};

use feedforward::FeedForward;
use inputlayer::InputLayer;
use outputlayer::OutputLayer;

pub struct Neuron {
    pub weights: Vec<Float>,
    bias: Float,
}

pub enum InitType {
    Xavier,
}

pub enum ActivationFunction {
    Sigmoid,
    Softmax,
}

pub enum Layer {
    InputLayer(InputLayer),
    FeedForward(FeedForward),
    OutputLayer(OutputLayer),
}

impl LayerTrait for Layer {
    fn forward(&mut self, input: Vec<Float>) {
        match self {
            Layer::InputLayer(layer) => layer.forward(input),
            Layer::FeedForward(layer) => layer.forward(input),
            Layer::OutputLayer(layer) => layer.forward(input),
        }
    }

    fn backward(
        &mut self,
        a: &Vec<Float>,
        layer_change: &mut LayerChange,
        error_input: &Vec<Float>,
        weights: Vec<Vec<Float>>,
        eta: Float,
    ) -> (Vec<Float>, Vec<Vec<Float>>) {
        match self {
            Layer::InputLayer(layer) => {
                (*layer).backward(a, layer_change, error_input, weights, eta)
            }
            Layer::FeedForward(layer) => {
                (*layer).backward(a, layer_change, error_input, weights, eta)
            }
            Layer::OutputLayer(layer) => {
                (*layer).backward(a, layer_change, error_input, weights, eta)
            }
        }
    }

    fn get_len(&self) -> usize {
        match self {
            Layer::InputLayer(layer) => (*layer).get_len(),
            Layer::FeedForward(layer) => (*layer).get_len(),
            Layer::OutputLayer(layer) => (*layer).get_len(),
        }
    }

    fn get_output(&self) -> Vec<Float> {
        match self {
            Layer::InputLayer(layer) => (*layer).get_output(),
            Layer::FeedForward(layer) => (*layer).get_output(),
            Layer::OutputLayer(layer) => (*layer).get_output(),
        }
    }

    fn get_z_values(&self) -> Vec<Float> {
        match self {
            Layer::InputLayer(layer) => (*layer).get_z_values(),
            Layer::FeedForward(layer) => (*layer).get_z_values(),
            Layer::OutputLayer(layer) => (*layer).get_z_values(),
        }
    }

    fn get_layer_change(&self) -> LayerChange {
        match self {
            Layer::InputLayer(layer) => (*layer).get_layer_change(),
            Layer::FeedForward(layer) => (*layer).get_layer_change(),
            Layer::OutputLayer(layer) => (*layer).get_layer_change(),
        }
    }
    fn update(&mut self, changes: &LayerChange, mini_batch_size: usize) {
        match self {
            Layer::InputLayer(layer) => (*layer).update(changes, mini_batch_size),
            Layer::FeedForward(layer) => (*layer).update(changes, mini_batch_size),
            Layer::OutputLayer(layer) => (*layer).update(changes, mini_batch_size),
        }
    }
}

pub trait LayerTrait {
    fn forward(&mut self, input: Vec<Float>);
    fn backward(
        &mut self,
        a: &Vec<Float>,
        layer_change: &mut LayerChange,
        error_input: &Vec<Float>,
        weights: Vec<Vec<Float>>,
        eta: Float,
    ) -> (Vec<Float>, Vec<Vec<Float>>);
    fn get_len(&self) -> usize;
    fn get_z_values(&self) -> Vec<Float>;
    fn get_output(&self) -> Vec<Float>;
    fn get_layer_change(&self) -> LayerChange;
    fn update(&mut self, changes: &LayerChange, mini_batch_size: usize);
}

impl Neuron {
    pub fn new(weights: Vec<Float>, bias: Float) -> Self {
        Neuron { weights, bias }
    }

    pub fn update(&mut self, neuron_change: &NeuronChange, mini_batch_size: usize) {
        update_bias(&mut self.bias, neuron_change.bias, mini_batch_size);
        update_weights(
            &mut self.weights,
            neuron_change.weights.clone(),
            mini_batch_size,
        );
    }
}

#[macro_export]
macro_rules! feedforward {
    ($init_type:expr, $activation_function:expr, $input_size:expr, $next_layer_size:expr, $length:expr) => {
        network::layer::Layer::FeedForward(network::layer::feedforward::FeedForward::new(
            $init_type,
            $activation_function,
            $input_size,
            $next_layer_size,
            $length,
        ))
    };
}

#[macro_export]
macro_rules! input {
    ($length:expr) => {
        network::layer::Layer::InputLayer(network::layer::inputlayer::InputLayer::new($length))
    };
}

#[macro_export]
macro_rules! output {
    ($init_type:expr, $activation_function:expr, $input_size:expr, $length:expr) => {
        network::layer::Layer::OutputLayer(network::layer::outputlayer::OutputLayer::new(
            $init_type,
            $activation_function,
            $input_size,
            $length,
        ))
    };
}

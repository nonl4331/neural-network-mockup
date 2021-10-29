pub mod feedforward;
pub mod inputlayer;
pub mod outputlayer;

use crate::network::{change::LayerChange, Float};

use {feedforward::FeedForward, inputlayer::InputLayer, outputlayer::OutputLayer};

pub enum Layer {
    FeedForward(FeedForward),
    InputLayer(InputLayer),
    OutputLayer(OutputLayer),
}

impl LayerTrait for Layer {
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

    fn empty_layer_change(&self) -> LayerChange {
        match self {
            Layer::FeedForward(layer) => (*layer).empty_layer_change(),
            Layer::InputLayer(layer) => (*layer).empty_layer_change(),
            Layer::OutputLayer(layer) => (*layer).empty_layer_change(),
        }
    }

    fn forward(&mut self, input: Vec<Float>) {
        match self {
            Layer::FeedForward(layer) => layer.forward(input),
            Layer::InputLayer(layer) => layer.forward(input),
            Layer::OutputLayer(layer) => layer.forward(input),
        }
    }

    fn last_output(&self) -> Vec<Float> {
        match self {
            Layer::FeedForward(layer) => (*layer).last_output(),
            Layer::InputLayer(layer) => (*layer).last_output(),
            Layer::OutputLayer(layer) => (*layer).last_output(),
        }
    }

    fn last_z_values(&self) -> Vec<Float> {
        match self {
            Layer::FeedForward(layer) => (*layer).last_z_values(),
            Layer::InputLayer(layer) => (*layer).last_z_values(),
            Layer::OutputLayer(layer) => (*layer).last_z_values(),
        }
    }

    fn neuron_count(&self) -> usize {
        match self {
            Layer::FeedForward(layer) => (*layer).neuron_count(),
            Layer::InputLayer(layer) => (*layer).neuron_count(),
            Layer::OutputLayer(layer) => (*layer).neuron_count(),
        }
    }

    fn update(&mut self, changes: &LayerChange, mini_batch_size: usize) {
        match self {
            Layer::FeedForward(layer) => (*layer).update(changes, mini_batch_size),
            Layer::InputLayer(layer) => (*layer).update(changes, mini_batch_size),
            Layer::OutputLayer(layer) => (*layer).update(changes, mini_batch_size),
        }
    }
}

pub trait LayerTrait {
    fn backward(
        &mut self,
        a: &Vec<Float>,
        layer_change: &mut LayerChange,
        error_input: &Vec<Float>,
        weights: Vec<Vec<Float>>,
        learning_rate: Float,
    ) -> (Vec<Float>, Vec<Vec<Float>>);
    fn empty_layer_change(&self) -> LayerChange;
    fn forward(&mut self, input: Vec<Float>);
    fn last_output(&self) -> Vec<Float>;
    fn last_z_values(&self) -> Vec<Float>;
    fn neuron_count(&self) -> usize;
    fn update(&mut self, changes: &LayerChange, mini_batch_size: usize);
}

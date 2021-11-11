pub mod feedforward;
pub mod inputlayer;
pub mod outputlayer;

use crate::network::{Float, Regularisation};

use {feedforward::FeedForward, inputlayer::InputLayer, outputlayer::OutputLayer};

pub enum Layer {
    FeedForward(FeedForward),
    InputLayer(InputLayer),
    OutputLayer(OutputLayer),
}

impl LayerTrait for Layer {
    fn backward(
        &mut self,
        a: &[Float],
        error_input: &[Float],
        weights: (Vec<Float>, [usize; 2]),
    ) -> (Vec<Float>, Vec<Float>, [usize; 2]) {
        match self {
            Layer::InputLayer(layer) => (*layer).backward(a, error_input, weights),
            Layer::FeedForward(layer) => (*layer).backward(a, error_input, weights),
            Layer::OutputLayer(layer) => (*layer).backward(a, error_input, weights),
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

    fn update(
        &mut self,
        learning_rate: Float,
        mini_batch_size: usize,
        regularisation: &Regularisation,
    ) {
        match self {
            Layer::FeedForward(layer) => {
                (*layer).update(learning_rate, mini_batch_size, regularisation)
            }
            Layer::InputLayer(layer) => {
                (*layer).update(learning_rate, mini_batch_size, regularisation)
            }
            Layer::OutputLayer(layer) => {
                (*layer).update(learning_rate, mini_batch_size, regularisation)
            }
        }
    }

    fn update_change(&mut self, errors: &[Float], a: &[Float]) {
        match self {
            Layer::FeedForward(layer) => (*layer).update_change(errors, a),
            Layer::InputLayer(layer) => (*layer).update_change(errors, a),
            Layer::OutputLayer(layer) => (*layer).update_change(errors, a),
        }
    }
}

pub trait LayerTrait {
    fn backward(
        &mut self,
        a: &[Float],
        error_input: &[Float],
        weights: (Vec<Float>, [usize; 2]),
        // errors, weights, dimensions_weights
    ) -> (Vec<Float>, Vec<Float>, [usize; 2]);
    fn forward(&mut self, input: Vec<Float>);
    fn last_output(&self) -> Vec<Float>;
    fn last_z_values(&self) -> Vec<Float>;
    fn update(
        &mut self,
        learning_rate: Float,
        mini_batch_size: usize,
        regularisation: &Regularisation,
    );
    fn update_change(&mut self, errors: &[Float], a: &[Float]);
}

use crate::network::{change::LayerChange, Float, Regularisation};

use super::LayerTrait;

pub struct InputLayer {
    length: usize,
    output: Vec<Float>,
}

impl LayerTrait for InputLayer {
    fn backward(
        &mut self,
        _: &Vec<Float>,
        _: &mut LayerChange,
        error_input: &Vec<Float>,
        weights: Vec<Vec<Float>>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        (error_input.clone(), weights)
    }

    fn empty_layer_change(&self) -> LayerChange {
        LayerChange::None
    }

    fn forward(&mut self, input: Vec<Float>) {
        assert_eq!(self.length, input.len());

        self.output = input;
    }

    fn last_output(&self) -> Vec<Float> {
        self.output.clone()
    }

    fn last_z_values(&self) -> Vec<Float> {
        self.output.clone()
    }

    fn neuron_count(&self) -> usize {
        self.length
    }

    fn update(&mut self, _: &LayerChange, _: Float, _: usize, _: &Regularisation) {}
}

impl InputLayer {
    pub fn new(length: usize) -> Self {
        InputLayer {
            length,
            output: Vec::new(),
        }
    }
}

#[macro_export]
macro_rules! input {
    ($length:expr) => {
        network::layer::Layer::InputLayer(network::layer::inputlayer::InputLayer::new($length))
    };
}

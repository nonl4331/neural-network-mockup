use crate::network::change::LayerChange;
use crate::network::{layer::LayerTrait, utility::Float};

pub struct InputLayer {
    length: usize,
    output: Vec<Float>,
}

impl LayerTrait for InputLayer {
    fn forward(&mut self, input: Vec<Float>) {
        assert_eq!(self.length, input.len());

        self.output = input;
    }
    fn backward(
        &mut self,
        a: &Vec<Float>,
        layer_change: &mut LayerChange,
        error_input: &Vec<Float>,
        weights: Vec<Vec<Float>>,
        eta: Float,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        (error_input.clone(), weights)
    }
    fn get_len(&self) -> usize {
        self.length
    }
    fn get_output(&self) -> Vec<Float> {
        self.output.clone()
    }
    fn get_z_values(&self) -> Vec<Float> {
        self.output.clone()
    }
    fn get_layer_change(&self) -> LayerChange {
        LayerChange::None
    }
    fn update(&mut self, changes: &LayerChange, mini_batch_size: usize) {}
}

impl InputLayer {
    pub fn new(length: usize) -> Self {
        InputLayer {
            length,
            output: Vec::new(),
        }
    }
}

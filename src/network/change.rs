use crate::network::Float;

pub struct FeedForwardChange {
    pub weights: Vec<Float>,
    pub biases: Vec<Float>,
}

impl FeedForwardChange {
    pub fn new(weight_dimensions: &[usize; 2]) -> Self {
        let weights = vec![0.0; weight_dimensions[0] * weight_dimensions[1]];
        FeedForwardChange {
            weights,
            biases: vec![0.0; weight_dimensions[0]],
        }
    }
}

pub struct OutputLayerChange {
    pub weights: Vec<Float>,
    pub biases: Vec<Float>,
}

impl OutputLayerChange {
    pub fn new(weight_dimensions: &[usize; 2]) -> Self {
        let weights = vec![0.0; weight_dimensions[0] * weight_dimensions[1]];
        OutputLayerChange {
            weights,
            biases: vec![0.0; weight_dimensions[0]],
        }
    }
}

use crate::network::Float;

pub enum ActivationFunction {
    Sigmoid,
    Softmax,
}

pub fn sigmoid(value: Float) -> Float {
    1.0 / (1.0 + (-value).exp())
}

pub fn d_sigmoid(value: Float) -> Float {
    sigmoid(value) * (1.0 - sigmoid(value))
}

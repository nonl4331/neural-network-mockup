use crate::network::Float;

pub enum ActivationFunction {
	Sigmoid,
	Softmax,
}

impl ActivationFunction {
	pub fn derivative(&self, value: Float) -> Float {
		match self {
			ActivationFunction::Sigmoid => d_sigmoid(value),
			ActivationFunction::Softmax => {
				unimplemented!()
			}
		}
	}

	pub fn evaluate(&self, value: Float) -> Float {
		match self {
			ActivationFunction::Sigmoid => sigmoid(value),
			ActivationFunction::Softmax => {
				unimplemented!()
			}
		}
	}
}

fn d_sigmoid(value: Float) -> Float {
	sigmoid(value) * (1.0 - sigmoid(value))
}

fn sigmoid(value: Float) -> Float {
	1.0 / (1.0 + (-value).exp())
}

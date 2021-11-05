use crate::network::{ActivationFunction, Float};

pub enum CostFunction {
    CrossEntropy,
    Quadratic,
    LogLikelyhood,
}

impl CostFunction {
    pub fn derivative(&self, value: Float, expected_value: Float) -> Float {
        match self {
            CostFunction::Quadratic => d_quadratic_cost(value, expected_value),
            CostFunction::CrossEntropy => {
                unimplemented!()
            }
            CostFunction::LogLikelyhood => {
                unimplemented!()
            }
        }
    }

    pub fn c_dz(
        &self,
        activation_function: &ActivationFunction,
        output: &Float,
        expected_value: &Float,
        z: &Float,
    ) -> Float {
        match self {
            CostFunction::Quadratic => {
                (output - expected_value) * activation_function.derivative(*z)
            }
            CostFunction::CrossEntropy => match activation_function {
                ActivationFunction::Sigmoid => output - expected_value,
                ActivationFunction::Softmax => {
                    unimplemented!()
                }
            },
            CostFunction::LogLikelyhood => match activation_function {
                ActivationFunction::Sigmoid => {
                    unimplemented!()
                }
                ActivationFunction::Softmax => output - expected_value,
            },
        }
    }
}

pub enum Regularisation {
    L1(Float),
    L2(Float),
    None,
}

fn d_quadratic_cost(value: Float, expected_value: Float) -> Float {
    value - expected_value
}

use crate::network::Float;

pub enum CostFunction {
    CrossEntropy,
    Quadratic,
}

impl CostFunction {
    pub fn derivative(&self, value: Float, expected_value: Float) -> Float {
        match self {
            CostFunction::Quadratic => d_quadratic_cost(value, expected_value),
            CostFunction::CrossEntropy => {
                unimplemented!()
            }
        }
    }
}

pub fn d_quadratic_cost(value: Float, expected_value: Float) -> Float {
    value - expected_value
}

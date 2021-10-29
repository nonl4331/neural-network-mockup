use crate::network::Float;

pub enum CostFunction {
    Quadratic,
    CrossEntropy,
}

pub fn d_quadratic_cost(value: Float, expected_value: Float) -> Float {
    value - expected_value
}

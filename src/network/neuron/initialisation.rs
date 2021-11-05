use crate::network::Float;

use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

pub enum InitType {
    He,
    NormalisedXavier,
    Xavier,
}

impl InitType {
    pub fn generate_weight(&self, in_num: usize, self_num: usize) -> Float {
        match self {
            InitType::He => he_init(in_num),
            InitType::NormalisedXavier => normalised_xavier_init(in_num, self_num),
            InitType::Xavier => xavier_init(in_num),
        }
    }
}

fn he_init(in_num: usize) -> Float {
    let mut rng = thread_rng();
    let std = (2.0 / in_num as Float).sqrt();
    let normal = Normal::new(0.0, std).unwrap();
    normal.sample(&mut rng)
}

fn normalised_xavier_init(in_num: usize, self_num: usize) -> Float {
    let mut rng = thread_rng();
    let val = (6.0 / (in_num + self_num) as Float).sqrt();
    rng.gen_range((-val)..val)
}

fn xavier_init(in_num: usize) -> Float {
    let mut rng = thread_rng();
    let val = (1.0 / in_num as Float).sqrt();
    rng.gen_range((-val)..val)
}

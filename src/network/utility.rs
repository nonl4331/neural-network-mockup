use blas::{saxpy, sgemv, sger, sscal};

pub type Float = f32;

pub fn hadamard_product(a: &[Float], b: &[Float]) -> Vec<Float> {
	a.iter().zip(b).map(|(&a, b)| a * b).collect()
}

/// max() over a slice of floats, gets the index of a largest value
pub fn max_index(nets: &[Float]) -> usize {
	let mut max = Float::NEG_INFINITY;
	let mut index = 0;
	for (i, n) in nets.iter().enumerate() {
		if *n > max {
			index = i;
			max = *n;
		}
	}
	index
}

// performs c += a * b
pub fn matrix_vec_multiply_add(a: &[Float], b: &[Float], c: &mut [Float], dim: &[usize; 2]) {
	unsafe {
		sgemv(
			b'N',
			dim[0] as i32,
			dim[1] as i32,
			1.0,
			a,
			dim[0] as i32,
			b,
			1,
			1.0,
			c,
			1,
		);
	}
}

// performs c += a * b^T
pub fn outer_product_add(a: &[Float], b: &[Float], c: &mut [Float]) {
	unsafe {
		sger(
			a.len() as i32,
			b.len() as i32,
			1.0,
			a,
			1,
			b,
			1,
			c,
			a.len() as i32,
		);
	}
}

// performs base_matrix += multiplier * matrix
pub fn plus_equals_matrix_multiplied(
	base_matrix: &mut [Float],
	multiplier: Float,
	matrix: &[Float],
) {
	assert_eq!(base_matrix.len(), matrix.len());
	unsafe {
		saxpy(
			base_matrix.len() as i32,
			multiplier,
			matrix,
			1,
			base_matrix,
			1,
		);
	}
}

// performs result = matrix^T * vector
pub fn transpose_matrix_multiply_vec(
	matrix: &[Float],
	vector: &[Float],
	dim: [usize; 2],
	result: &mut Vec<Float>,
) {
	result.reserve_exact(dim[1]);
	unsafe {
		sgemv(
			b'T',
			dim[0] as i32,
			dim[1] as i32,
			1.0,
			matrix,
			dim[0] as i32,
			vector,
			1,
			0.0,
			result,
			1,
		);
		result.set_len(dim[1]);
	}
}

// performs a *= multiplier;
pub fn scale_elements(a: &mut [Float], multiplier: Float) {
	unsafe {
		sscal(a.len() as i32, multiplier, a, 1);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn gemv() {
		// note column major
		let a = [3.2, 5.3, 0.0, 1.2, -0.2, -1.1];

		let dim = [3, 2];

		let b = [1.3, -0.5];

		let mut c = [1.3, -2.3, -0.1];

		matrix_vec_multiply_add(&a, &b, &mut c, &dim);

		assert!(
			c[0] > 4.85999
				&& c[0] < 4.86001
				&& c[1] > 4.68999
				&& c[1] < 4.69001
				&& c[2] > 0.44999
				&& c[2] < 0.45001
		);
	}

	#[test]
	fn saxpy() {
		let mut a = [3.2, -0.2, 1.2, 4.5];
		let multiplier = -0.5;
		let b = [4.0, 1.2, -6.0, -5.0];

		plus_equals_matrix_multiplied(&mut a, multiplier, &b);

		assert_eq!(a, [1.2, -0.8, 4.2, 7.0]);
	}

	#[test]
	fn outer_product() {
		let a = [1.0, 2.0, 3.0];
		let b = [4.0, 5.0];
		let mut c = [0.2, -1.0, -0.5, 4.3, 5.0, 0.7];

		outer_product_add(&a, &b, &mut c);

		assert_eq!(c, [4.2, 7.0, 11.5, 9.3, 15.0, 15.7]);
	}

	#[test]
	fn sgemv_transpose_no_add() {
		let a = [3.2, 5.7, 1.2, -6.0, -0.3, 9.5];
		let dim = [3, 2];
		let b = [-0.5, 5.5, 1.3];
		let mut res = Vec::new();
		transpose_matrix_multiply_vec(&a, &b, dim, &mut res);

		assert!(res[0] > 31.3099 && res[0] < 32.31001 && res[1] > 13.6999 && res[1] < 13.70001);
	}
}

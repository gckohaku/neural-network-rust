use std::ops;
use crate::matrix::Matrix;

// Matrix /= f64
impl ops::DivAssign<f64> for Matrix {
	fn div_assign(&mut self, rhs: f64) {
		if rhs == 0.0 {
			panic!("The divisor must not be zero.");
		}

		*self *= 1.0 / rhs;
	}	
}

// Matrix / f64
impl ops::Div<f64> for Matrix {
	type Output = Result<Matrix, String>;

	fn div(self, rhs: f64) -> Self::Output {
		if rhs == 0.0 {
			return Err("The divisor must not be zero.".to_string())
		}

		let mut result = self.clone();
		result /= rhs;
		Ok(result)
	}
}

// &Matrix / f64
impl ops::Div<f64> for &Matrix {
	type Output = Result<Matrix, String>;

	fn div(self, rhs: f64) -> Self::Output {
		if rhs == 0.0 {
			return Err("The divisor must not be zero.".to_string())
		}

		let mut result = self.clone();
		result /= rhs;
		Ok(result)
	}
}
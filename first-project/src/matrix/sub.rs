use std::ops;

use crate::matrix::Matrix;

// Matrix -= Matrix
impl ops::SubAssign<Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: Matrix) {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            // for i in 0..self.rows {
            // 	for j in 0..self.cols {
            // 		let diff = self.get(i, j).unwrap() - rhs.get(i, j).unwrap();
            // 		self.set(i, j, diff).unwrap();
            // 	}
            // }
            *self += rhs * -1.0;
        } else {
            panic!("Matrices must have the same size for subtraction");
        }
    }
}

// Matrix -= &Matrix
impl ops::SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            *self += rhs * -1.0;
        } else {
            panic!("Matrices must have the same size for subtraction");
        }
    }
}

// Matrix -= Result<Matrix, String>
impl ops::SubAssign<Result<Matrix, String>> for Matrix {
	fn sub_assign(&mut self, rhs: Result<Matrix, String>) {
		match rhs {
			Ok(matrix) => self.sub_assign(matrix),
			Err(e) => panic!("Error subtracting matrix: {}", e),
		}
	}
}

// Matrix - Matrix
impl ops::Sub<Matrix> for Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            let mut result = self.clone();
            result -= rhs;
            Ok(result)
        } else {
            panic!("Matrices must have the same size for subtraction")
        }
    }
}

// &Matrix - &Matrix
impl ops::Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            let mut result = self.clone();
            result -= rhs;
            Ok(result)
        } else {
            Err("Matrices must have the same size for subtraction".to_string())
        }
    }
}

// Matrix + Result<Matrix, String>
impl ops::Sub<Result<Matrix, String>> for Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, rhs: Result<Matrix, String>) -> Self::Output {
        match rhs {
            Ok(matrix) => self - matrix,
            Err(e) => Err(e),
        }
    }
}

// Result<Matrix, String> + Matrix
impl ops::Sub<Matrix> for Result<Matrix, String> {
    type Output = Result<Matrix, String>;  

    fn sub(self, rhs: Matrix) -> Self::Output {
        match self {
            Ok(matrix) => matrix - rhs,
            Err(e) => Err(e),
        }
    }
}

// &Matrix + Result<Matrix, String>
impl ops::Sub<Result<Matrix, String>> for &Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, rhs: Result<Matrix, String>) -> Self::Output {
        match rhs {
            Ok(matrix) => self - &matrix,
            Err(e) => Err(e),
        }
    }
}

// Result<Matrix, String> + &Matrix
impl ops::Sub<&Matrix> for Result<Matrix, String> {
    type Output = Result<Matrix, String>;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        match self {
            Ok(matrix) => &matrix - rhs,
            Err(e) => Err(e),
        }
    }
}
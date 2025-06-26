use std::{ops::{self, AddAssign}, result};

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn new_from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Matrix, String> {
        if data.len() == rows * cols {
            Ok(Matrix { rows, cols, data })
        } else {
            Err("Data length does not match specified dimensions".to_string())
        }
    }

    pub fn new_from_2dim_vec(data: Vec<Vec<f64>>) -> Result<Matrix, String> {
        if data.is_empty() || data.iter().any(|row| row.len() != data[0].len()) {
            return Err("All rows must have the same length".to_string());
        }
        let rows = data.len();
        let cols = data[0].len();
        let flat_data: Vec<f64> = data.into_iter().flatten().collect();
        Matrix::new_from_vec(rows, cols, flat_data)
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
            Ok(())
        } else {
            Err("Index out of bounds".to_string())
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.set(j, i, self.get(i, j).unwrap()).unwrap();
            }
        }
        transposed
    }
}

// Matrix += Matrix
impl ops::AddAssign<Matrix> for Matrix {
    fn add_assign(&mut self, rhs: Matrix) {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let sum = self.get(i, j).unwrap() + rhs.get(i, j).unwrap();
                    self.set(i, j, sum).unwrap();
                }
            }
        } else {
            panic!("Matrices must have the same size for addition");
        }
    }
}

// Matrix += &Matrix
impl ops::AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let sum = self.get(i, j).unwrap() + rhs.get(i, j).unwrap();
                    self.set(i, j, sum).unwrap();
                }
            }
        } else {
            panic!("Matrices must have the same size for addition");
        }
    }
}

// Matrix + Matrix
impl ops::Add<Matrix> for Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            let mut result = self.clone();
            result += rhs;
            Ok(result)
        } else {
            panic!("Matrices must have the same size for addition")
        }
    }
}

// &Matrix + &Matrix
impl ops::Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: &Matrix) -> Self::Output {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            let mut result = self.clone();
            result += rhs;
            Ok(result)
        } else {
            Err("Matrices must have the same size for addition".to_string())
        }
    }
}

// Matrix *= f64
impl ops::MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let product = self.get(i, j).unwrap() * scalar;
                self.set(i, j, product).unwrap();
            }
        }
    }
}

// Matrix *= &f64
impl ops::MulAssign<&f64> for Matrix {
    fn mul_assign(&mut self, scalar: &f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let product = self.get(i, j).unwrap() * scalar;
                self.set(i, j, product).unwrap();
            }
        }
    }
}

// Matrix * f64
impl ops::Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Self::Output {
        let mut result = self.clone();
        result *= scalar;
        result
    }
}

// &Matrix * f64
impl ops::Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Self::Output {
        let mut result = self.clone();
        result *= scalar;
        result
    }
}

// f64 * Matrix
impl ops::Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, matrix: Matrix) -> Self::Output {
        let mut result = matrix.clone();
        result *= self;
        result
    }
}

// f64 * &Matrix
impl ops::Mul<&Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, matrix: &Matrix) -> Self::Output {
        let mut result = matrix.clone();
        result *= self;
        result
    }
}

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

// Matrix * Matrix
// ベクトルモジュールを定義してから
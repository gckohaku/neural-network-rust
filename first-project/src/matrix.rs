mod add;

use std::{fmt::{Debug, Display}, ops, vec};

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

    pub fn new_and_fill(rows: usize, cols: usize, value: f64) -> Matrix {
        let mut matrix = Matrix::new(rows, cols);
        matrix.fill(value);
        matrix
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

    pub fn get_row(&self, row: usize) -> Result<Vec<f64>, String> {
        if row < self.rows {
            let start = row * self.cols;
            let end = start + self.cols;
            Ok(self.data[start..end].to_vec())
        } else {
            Err("Row index out of bounds".to_string())
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

    pub fn fill(&mut self, value: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, value).unwrap();
            }
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

    pub fn hadamard_assign(&mut self, other: &Matrix) -> Result<(), String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices must have the same size for Hadamard product".to_string());
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                let a = self.get(i, j).unwrap();
                let b = other.get(i, j).unwrap();
                self.set(i, j, a * b)?;
            }
        }
        Ok(())
    }

    pub fn hadamard(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices must have the same size for Hadamard product".to_string());
        }
		let mut result = self.clone();
		result.hadamard_assign(other)?;
		Ok(result)
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap();
        let precision = f.precision().unwrap_or(0);

        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:width$.precision$}", self.get(i, j).unwrap(), width = width, precision = precision)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl ops::Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        if row < self.rows && col < self.cols {
            &self.data[row * self.cols + col]
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        if row < self.rows && col < self.cols {
            &mut self.data[row * self.cols + col]
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl ops::Index<usize> for Matrix {
    type Output = [f64];

    fn index(&self, index: usize) -> &Self::Output {
        if index < self.rows {
            &self.data[index * self.cols..(index + 1) * self.cols]
        } else {
            panic!("Row index out of bounds");
        }
    }
}

impl ops::IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index < self.rows {
            &mut self.data[index * self.cols..(index + 1) * self.cols]
        } else {
            panic!("Row index out of bounds");
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

// Matrix *= Matrix
impl ops::MulAssign<Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: Matrix) {
        if self.cols == rhs.rows {
            let mut result = Matrix::new(self.rows, rhs.cols);
            for i in 0..self.rows {
                for j in 0..rhs.cols {
                    let mut sum = 0.0;
                    for k in 0..self.cols {
                        sum += self.get(i, k).unwrap() * rhs.get(k, j).unwrap();
                    }
                    result.set(i, j, sum).unwrap();
                }
            }
            *self = result;
        } else {
            panic!("Matrices must have compatible dimensions for multiplication");
        }
    }
}

// Matrix *= &Matrix
impl ops::MulAssign<&Matrix> for Matrix {
    fn mul_assign(&mut self, rhs: &Matrix) {
        if self.cols == rhs.rows {
            let mut result = Matrix::new(self.rows, rhs.cols);
            for i in 0..self.rows {
                for j in 0..rhs.cols {
                    let mut sum = 0.0;
                    for k in 0..self.cols {
                        sum += self.get(i, k).unwrap() * rhs.get(k, j).unwrap();
                    }
                    result.set(i, j, sum).unwrap();
                }
            }
            *self = result;
        } else {
            panic!("Matrices must have compatible dimensions for multiplication");
        }
    }
}

// Matrix * Matrix
impl ops::Mul<Matrix> for Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: Matrix) -> Self::Output {
        if self.cols == rhs.rows {
            let mut result = self.clone();
            result *= rhs;
            Ok(result)
        } else {
            Err("Matrices must have compatible dimensions for multiplication".to_string())
        }
    }
}

// &Matrix * &Matrix
impl ops::Mul<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.cols == rhs.rows {
            let mut result = self.clone();
            result *= rhs;
            Ok(result)
        } else {
            Err("Matrices must have compatible dimensions for multiplication".to_string())
        }
    }
}

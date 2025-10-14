use std::{
    cell::UnsafeCell,
    collections::HashMap,
    num::ParseIntError,
    ops,
    sync::{Arc, Mutex, mpsc},
    thread, time,
};

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{cpu_info, matrix::Matrix};

/* 実数との乗算 */
// Matrix *= f64
impl ops::MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let product = self[(i, j)] * scalar;
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
                let product = self[(i, j)] * scalar;
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

/* 行列積 */
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
            // let cpu_cores = *cpu_info::LOGICAL_CORES;
            // let arc_self = Arc::new(self.clone());
            // let arc_rhs = Arc::new(rhs.clone());

            let mut result = Matrix::new(self.rows, rhs.cols);

            let block_size = 4;

            let start = time::Instant::now();
            result
                .data
                .par_chunks_mut(rhs.cols * block_size)
                .enumerate()
                .for_each(|(chunk_index, result_block_slice)| {
                    let start_row_index = chunk_index * block_size;

                    for row_offset in 0..block_size {
                        let row_index = start_row_index + row_offset;
                        let result_row_slice = &mut result_block_slice
                            [(row_offset * rhs.cols)..((row_offset + 1) * rhs.cols)];

                        for k in 0..self.cols {
                            let self_ik = self.data[row_index * self.cols + k];
                            for col_index in 0..rhs.cols {
                                result_row_slice[col_index] +=
                                    self_ik * rhs.data[k * rhs.cols + col_index];
                            }
                        }
                    }
                });
            println!("actual calc time: {:?}", start.elapsed());

            *self = result;
        } else {
            panic!("Matrices must have compatible dimensions for multiplication");
        }
    }
}

// Matrix *= Result<Matrix, String>
impl ops::MulAssign<Result<Matrix, String>> for Matrix {
    fn mul_assign(&mut self, rhs: Result<Matrix, String>) {
        match rhs {
            Ok(matrix) => self.mul_assign(matrix),
            Err(e) => panic!("Error multiplying matrix: {}", e),
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

// Matrix * Result<Matrix, String>
impl ops::Mul<Result<Matrix, String>> for Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: Result<Matrix, String>) -> Self::Output {
        match rhs {
            Ok(matrix) => self * matrix,
            Err(e) => Err(e),
        }
    }
}

// Result<Matrix, String> * Matrix
impl ops::Mul<Matrix> for Result<Matrix, String> {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: Matrix) -> Self::Output {
        match self {
            Ok(matrix) => matrix * rhs,
            Err(e) => Err(e),
        }
    }
}

// &Matrix * Result<Matrix, String>
impl ops::Mul<Result<Matrix, String>> for &Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: Result<Matrix, String>) -> Self::Output {
        match rhs {
            Ok(matrix) => self * &matrix,
            Err(e) => Err(e),
        }
    }
}

// Result<Matrix, String> * &Matrix
impl ops::Mul<&Matrix> for Result<Matrix, String> {
    type Output = Result<Matrix, String>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        match self {
            Ok(matrix) => &matrix * rhs,
            Err(e) => Err(e),
        }
    }
}

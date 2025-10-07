use std::{
    ops,
    sync::{Arc, Mutex},
    thread,
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
            let cpu_cores = *cpu_info::LOGICAL_CORES;
            let result = Matrix::new(self.rows, rhs.cols);
            let arc_result = Arc::new(result);
            let arc_self = Arc::new(self.clone());
            let arc_rhs = Arc::new(rhs.clone());

            // println!("start:\n  self: row -> {}, col -> {}\n  rhs: row -> {}, col -> {}", self.rows, self.cols, rhs.rows, rhs.cols);

            let mut handles: Vec<thread::JoinHandle<()>> = vec![];
            for offset in 0..cpu_cores {
                let mc_self = Arc::clone(&arc_self);
                let mc_rhs = Arc::clone(&arc_rhs);
                let results: Vec<Vec<f64>> = vec![];

                let mc_result = Arc::clone(&arc_result);

                let thread = thread::spawn({
                    move || {
                        let mut index = offset;

                        while index < mc_self.rows * mc_rhs.cols {
                            let cell_row = index / mc_rhs.cols;
                            let cell_col = index % mc_rhs.cols;

                            let mut sum = 0.0;
                            for k in 0..mc_self.cols {
                                sum += mc_self[(cell_row, k)] * mc_rhs[(k, cell_col)];
                            }

                            mc_result.set(cell_row, cell_col, sum).unwrap();

                            index += cpu_cores;
                        }
                    }
                });

                handles.push(thread);
            }

            // println!("handle value: {}", handles.len());

            for handle in handles {
                handle.join().unwrap();
            }

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

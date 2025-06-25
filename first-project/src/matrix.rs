use std::{ops};

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

impl ops::Add<Matrix> for Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows == rhs.rows && self.cols == rhs.cols {
            let mut result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let sum = self.get(i, j).unwrap() + rhs.get(i, j).unwrap();
                    result.set(i, j, sum).unwrap();
                }
            }
            Ok(result)
        } else {
            Err("Matrices must have the same size for addition".to_string())
        }
    }
}

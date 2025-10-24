mod add;
mod div;
mod mul;
mod sub;

use std::{
    fmt::{Debug, Display},
    ops, vec,
};

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
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
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

    pub fn add_cell(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] += value;
            Ok(())
        } else {
            Err("Index out of bounds".to_string())
        }
    }

    pub fn fill(&mut self, value: f64) {
        // for i in 0..self.rows {
        //     for j in 0..self.cols {
        //         self.set(i, j, value).unwrap();
        //     }
        // }
        self.data = vec![value; self.rows * self.cols];
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
        if self.rows % other.rows != 0 || self.cols % other.cols != 0 {
            return Err(
                "Matrices must have the same size or able to broadcast for Hadamard product"
                    .to_string(),
            );
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                let a = self.get(i, j).unwrap();
                let b = other.get(i % other.rows, j % other.cols).unwrap();
                self.set(i, j, a * b)?;
            }
        }
        Ok(())
    }

    pub fn hadamard(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows % other.rows != 0 || self.cols % other.cols != 0 {
            return Err(
                "Matrices must have the same size or able to broadcast for Hadamard product"
                    .to_string(),
            );
        }
        let mut result = self.clone();
        result.hadamard_assign(other)?;
        Ok(result)
    }

    pub fn hadamard_function_assign(&mut self, function: fn(&f64) -> f64) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let element = self.get(row, col).unwrap();
                self.set(row, col, function(&element)).unwrap();
            }
        }
    }

    pub fn hadamard_function(&mut self, function: fn(&f64) -> f64) -> Matrix {
        let mut ret = self.clone();
        ret.hadamard_function_assign(function);
        ret
    }

    pub fn sum_all_elements(&mut self) -> f64 {
        self.data.iter().sum::<f64>()
    }

    pub fn sum_row_elements(&mut self, row: usize) -> f64 {
        self.get_row(row).unwrap().iter().sum::<f64>()
    }

    pub fn change_row_size(&mut self, row_size: usize) {
        self.rows = row_size;
        self.data.resize(row_size * self.cols, 0.0);
    }

    pub fn mean_cols(&mut self) -> Matrix {
        let mut result = Matrix::new(1, self.cols);

        for col in 0..self.cols {
            let mut col_sum = 0.0;
            for row in 0..self.rows {
                col_sum += self[(row, col)];
            }
            result.set(0, col, col_sum / self.rows as f64).unwrap();
        }

        result
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(0);
        let precision = f.precision().unwrap_or(0);

        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(
                    f,
                    "{:width$.precision$}",
                    self.get(i, j).unwrap(),
                    width = width,
                    precision = precision
                )?;
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

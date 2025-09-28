use crate::matrix::Matrix;
use std::ops;

// Matrix += Matrix
impl ops::AddAssign<Matrix> for Matrix {
    fn add_assign(&mut self, rhs: Matrix) {
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for addition");
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                let sum = self[(i, j)] + rhs[(i % rhs.rows, j % rhs.cols)];
                self.set(i, j, sum).unwrap();
            }
        }
    }
}

// Matrix += &Matrix
impl ops::AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for addition");
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                let sum = self[(i, j)] + rhs[(i % rhs.rows, j % rhs.cols)];
                self.set(i, j, sum).unwrap();
            }
        }
    }
}

// Matrix += Result<Matrix, String>
impl ops::AddAssign<Result<Matrix, String>> for Matrix {
    fn add_assign(&mut self, rhs: Result<Matrix, String>) {
        match rhs {
            Ok(matrix) => self.add_assign(matrix),
            Err(e) => panic!("Error adding matrix: {}", e),
        }
    }
}

// Matrix + Matrix
impl ops::Add<Matrix> for Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for addition");
        }

        let mut result = self.clone();
        result += rhs;
        Ok(result)
    }
}

// &Matrix + &Matrix
impl ops::Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: &Matrix) -> Self::Output {
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for addition");
        }

        let mut result = self.clone();
        result += rhs;
        Ok(result)
    }
}

// Matrix + Result<Matrix, String>
impl ops::Add<Result<Matrix, String>> for Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: Result<Matrix, String>) -> Self::Output {
        match rhs {
            Ok(matrix) => self + matrix,
            Err(e) => Err(e),
        }
    }
}

// Result<Matrix, String> + Matrix
impl ops::Add<Matrix> for Result<Matrix, String> {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: Matrix) -> Self::Output {
        match self {
            Ok(matrix) => matrix + rhs,
            Err(e) => Err(e),
        }
    }
}

// &Matrix + Result<Matrix, String>
impl ops::Add<Result<Matrix, String>> for &Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: Result<Matrix, String>) -> Self::Output {
        match rhs {
            Ok(matrix) => self + &matrix,
            Err(e) => Err(e),
        }
    }
}

// Result<Matrix, String> + &Matrix
impl ops::Add<&Matrix> for Result<Matrix, String> {
    type Output = Result<Matrix, String>;

    fn add(self, rhs: &Matrix) -> Self::Output {
        match self {
            Ok(matrix) => &matrix + rhs,
            Err(e) => Err(e),
        }
    }
}

use std::ops;

use crate::matrix::Matrix;

// Matrix -= Matrix
impl ops::SubAssign<Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: Matrix) {
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for subtraction");
        }

        *self += rhs * -1.0;
    }
}

// Matrix -= &Matrix
impl ops::SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for subtraction");
        }

        *self += rhs * -1.0;
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
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for subtraction");
        }

        let mut result = self.clone();
        result -= rhs;
        Ok(result)
    }
}

// &Matrix - &Matrix
impl ops::Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        if self.rows % rhs.rows != 0 || self.cols % rhs.cols != 0 {
            panic!("Matrices must have the same size or able to broadcast for subtraction");
        }

        let mut result = self.clone();
        result -= rhs;
        Ok(result)
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

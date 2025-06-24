use matrix::*;

fn main() {
	let mut mat: Matrix = matrix::Matrix::new(2, 3);

	println!("Matrix created with {} rows and {} columns", mat.rows, mat.cols);
}

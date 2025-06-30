mod matrix;

use matrix::*;

fn main() {
	let mat_a: Matrix = Matrix::new_from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
	let mat_b: Matrix = Matrix::new_from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
	
	println!("{:8.2}\n{:8.2}\n{:8.2}", &mat_a, &mat_b, (&mat_a * &mat_b).unwrap());
}
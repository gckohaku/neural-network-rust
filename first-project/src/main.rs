mod matrix;

use matrix::*;

fn main() {
	let mat_a: Matrix = Matrix::new_from_vec(2, 3, vec![3.0, 1.0, 2.0, 7.0, 5.0, 4.0]).unwrap();
	let mat_b: Matrix = Matrix::new_from_vec(2, 3, vec![3.0, 4.0, 7.0, 6.0, 4.0, 3.0]).unwrap();
	
	println!("{:?}, {:?}", &mat_a + &mat_b, &mat_a - &mat_b);
}
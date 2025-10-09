use crate::{iris_nn::{calc_average_and_variance, calc_kurtosis, calc_skewness, iris_nn_process}, matrix::Matrix};

mod iris_nn;
mod matrix;
mod neural_network;
mod output_activation_type;
mod rand;
mod ron_data;
mod iris_normalization;
mod cpu_info;

fn main() {
    iris_nn_process();

    // iris_analyze();

    // let mut m1 = Matrix::new_from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    // let m2 =  Matrix::new_from_vec(3, 3, vec![3.0, 4.0, 6.0, 4.0, 2.0, 5.0, 3.0, 1.0, 4.0]).unwrap();

    // m1 *= &m2;

    // println!("{:7.1}", m1);
}

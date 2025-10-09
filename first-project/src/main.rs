use crate::{iris_nn::{calc_average_and_variance, calc_kurtosis, calc_skewness, iris_nn_process}, matrix::Matrix};

mod iris_nn;
mod matrix;
mod neural_network;
mod output_activation_type;
mod rand;
mod ron_data;
mod iris_normalization;
mod cpu_info;
mod matrix_mul_calc_speed;

fn main() {
    // iris_nn_process();

    // iris_analyze();

    matrix_mul_calc_speed::calc();
}

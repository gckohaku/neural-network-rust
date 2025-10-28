use crate::iris_nn::{calc_average_and_variance, calc_kurtosis, calc_skewness, iris_nn_process};

mod iris_nn;
mod iris_nn_mt;
mod matrix;
mod fully_connected_network;
mod output_activation_type;
mod rand;
mod ron_data;
mod iris_normalization;
mod matrix_mul_calc_speed;
mod constants;
mod neural_network_base;
mod neural_network_functions;
mod mnist_nn;

fn main() {
    iris_nn::iris_nn_process();

    // iris_analyze();
}

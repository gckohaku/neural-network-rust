use crate::{constants::MNIST_MT_CHUNK_SIZE, iris_nn::{calc_average_and_variance, calc_kurtosis, calc_skewness, iris_nn_process}, mnist_nn::mnist_process};

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
mod mnist_nn_mt;
mod utilities;

fn main() {
    // mnist_process();
    mnist_nn_mt::mnist_process(3, MNIST_MT_CHUNK_SIZE, 4, 50000, 5000, 5000);
}

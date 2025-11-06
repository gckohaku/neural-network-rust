use crate::{
    constants::MNIST_MT_CHUNK_SIZE,
    iris_nn::{calc_average_and_variance, calc_kurtosis, calc_skewness, iris_nn_process},
    mnist_nn::mnist_process,
};

mod constants;
mod fully_connected_network;
mod iris_nn;
mod iris_nn_mt;
mod iris_normalization;
mod matrix;
mod matrix_mul_calc_speed;
mod mnist_nn;
mod mnist_nn_mt;
mod neural_network_base;
mod neural_network_functions;
mod output_activation_type;
mod rand;
mod ron_data;
mod utilities;

fn main() {
    let epochs = 20;
    let learning_chunk_size = 250;
    let mini_batch_iteration = 16;
    let validation_iteration = 10;
    let test_iteration = 10;
    let training_max_value = 60000;
    let validation_max_value = 10000;
    let test_max_value = 0;
    let validation_chunk_size = 500;
    let test_chunk_size = 500;

    // mnist_process();
    mnist_nn_mt::mnist_process(epochs, learning_chunk_size, mini_batch_iteration, validation_iteration, test_iteration, training_max_value, validation_max_value, test_max_value, validation_chunk_size, test_chunk_size);
}

use crate::iris_nn::{calc_average_and_variance, iris_analyze};

mod iris_nn;
mod matrix;
mod neural_network;
mod output_activation_type;
mod rand;
mod ron_data;

fn main() {
    // iris_nn_process();

    iris_analyze();
}

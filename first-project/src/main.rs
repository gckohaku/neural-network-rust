use crate::iris_nn::{calc_average_and_variance, calc_kurtosis, calc_skewness, iris_nn_process};

mod iris_nn;
mod matrix;
mod neural_network;
mod output_activation_type;
mod rand;
mod ron_data;
mod iris_normalization;

fn main() {
    // iris_nn_process();

    // iris_analyze();

    let data = vec![3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 7.0];
    let (average, variance, unbiased ) = calc_average_and_variance(&data);
    let skewness = calc_skewness(&data, average, variance);
    let kurtosis = calc_kurtosis(&data, average, unbiased);

    println!("average : {}", average);
    println!("variance: {}", variance);
    println!("standard: {}", variance.sqrt());
    println!("skewness: {}", skewness);
    println!("kurtosis: {}", kurtosis);
}

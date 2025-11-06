use crate::matrix::Matrix;

pub fn relu(x: &f64) -> f64 {
    x.max(0.0)
}

pub fn differential_relu(x: &f64) -> f64 {
    if *x > 0.0 { 1.0 } else { 0.0 }
}

pub fn leaky_relu(x: &f64) -> f64 {
    if *x > 0.0 { *x } else { 0.01 * *x }
}

pub fn differential_leaky_relu(x: &f64) -> f64 {
    if *x > 0.0 { 1.0 } else { 0.01 }
}

pub fn sigmoid(x: &f64) -> f64 {
    ((x / 2.0).tanh() + 1.0) / 2.0
}

pub fn differential_sigmoid(x: &f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn softmax(z: &mut Matrix) {
    let nodes_exp_sum: f64 = z.data.iter().map(|x| x.exp()).sum();
    for i in 0..z.rows {
        for j in 0..z.cols {
            let before_normalize = z.get(i, j).unwrap();
            z.set(i, j, before_normalize.exp() / nodes_exp_sum).unwrap();
        }
    }
}

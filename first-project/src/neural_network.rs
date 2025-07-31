use std::fmt::Display;

use crate::{matrix::Matrix, rand::Rand};

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    nodes: Vec<Matrix>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    activations: Vec<fn(&f64) -> f64>,
    differential_activations: Vec<fn(&f64) -> f64>,
    deltas: Vec<Matrix>,
    use_softmax_flags: Vec<bool>,
}

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

pub fn softmax(z: &mut Matrix) {
    let nodes_exp_sum: f64 = z.data.iter().map(|x| x.exp()).sum();
    for i in 0 .. z.rows {
        for j in 0 .. z.cols {
            let before_normalize = z.get(i, j).unwrap();
            z.set(i, j, before_normalize.exp() / nodes_exp_sum).unwrap();
        }
    }
}

impl NeuralNetwork {
    pub fn new(nodes_values: Vec<usize>) -> Self {
        let mut nodes = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let activations: Vec<fn(&f64) -> f64> = Vec::new();
        let differential_activations: Vec<fn(&f64) -> f64> = Vec::new();
        let mut deltas = Vec::new();
        let use_softmax_flags = Vec::new();

        let mut r = Rand::new();

        for i in 0..nodes_values.len() {
            nodes.push(Matrix::new_and_fill(nodes_values[i], 1, 0.0));
            if i > 0 {
                let mut layer_weights = Matrix::new(nodes_values[i], nodes_values[i - 1]);
                for row in 0..layer_weights.rows {
                    for col in 0..layer_weights.cols {
                        layer_weights
                            .set(
                                row,
                                col,
                                r.normal(0.0, (2.0 / nodes_values[i - 1] as f64).sqrt()),
                            )
                            .unwrap();
                    }
                }
                weights.push(layer_weights);
                biases.push(Matrix::new_and_fill(nodes_values[i], 1, 0.0));
                deltas.push(Matrix::new_and_fill(nodes_values[i], 1, 0.0));
            }
        }

        Self {
            nodes,
            weights,
            biases,
            activations,
            differential_activations,
            deltas,
            use_softmax_flags,
        }
    }

    pub fn set_activations(&mut self, activations: &mut Vec<fn(&f64) -> f64>) {
        self.activations.clear();
        self.activations.append(activations);
    }

    pub fn set_differential_activation(&mut self, differentials: &mut Vec<fn(&f64) -> f64>) {
        self.differential_activations.clear();
        self.differential_activations.append(differentials);
    }

    pub fn set_use_softmax_flags(&mut self, flags: &mut Vec<bool>) {
        self.use_softmax_flags.clear();
        self.use_softmax_flags.append(flags);
    }

    pub fn view_status(&self) {
        println!("Neural Network Status:");
        println!("Nodes: {:?}", self.nodes);
        println!("Weights: {:?}", self.weights);
        println!("Biases: {:?}", self.biases);
        println!("Deltas: {:?}", self.deltas);
    }

    pub fn forward(&mut self, inputs: &Matrix) -> Result<(), String> {
        if inputs.rows != self.nodes[0].rows || inputs.cols != 1 {
            return Err(
                "Input dimensions do not match the first layer of the network.".to_string(),
            );
        }

        self.nodes[0] = inputs.clone();

        for i in 1..self.nodes.len() {
            let weighted_sum =
                (&self.weights[i - 1] * &self.nodes[i - 1] + &self.biases[i - 1]).unwrap();

            if self.use_softmax_flags[i - 1] {
                softmax(&mut self.nodes[i]);
            } else {
                for j in 0..self.nodes[i].rows {
                    self.nodes[i].set(j, 0, self.activations[i - 1](&weighted_sum[j][0]))?;
                }
            }
        }

        Ok(())
    }

    pub fn backward(&mut self, expected: &Matrix) -> Result<(), String> {
        if expected.rows != self.nodes.last().unwrap().rows || expected.cols != 1 {
            return Err(
                "Expected output dimensions do not match the last layer of the network."
                    .to_string(),
            );
        }

        // Backward pass logic would go here
        // This is a placeholder for now

        Ok(())
    }
}

impl Display for NeuralNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(10);
        let precision = f.precision().unwrap_or(0);

        for i in 0..self.nodes.len() {
            if i > 0 {
                writeln!(f, "Layer {}:", i)?;

                writeln!(f, "\nWeights:")?;
                write!(f, "{:width$.precision$} ", self.weights[i - 1])?;

                writeln!(f, "\nBiases:")?;
                write!(f, "{:width$.precision$} ", self.biases[i - 1])?;

                writeln!(f, "\nNodes:")?;
            } else {
                writeln!(f, "Input Layer:")?;
                writeln!(f, "Inputs:")?;
            }

            write!(
                f,
                "{:width$.precision$} ",
                self.nodes[i],
                width = width,
                precision = precision
            )?;

            writeln!(f)?;
        }
        Ok(())
    }
}

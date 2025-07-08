use std::fmt::Display;

use crate::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    nodes: Vec<Matrix>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    deltas: Vec<Matrix>,
}

fn relu(x: &f64) -> f64 {
    if *x > 0.0 { *x } else { 0.0 }
}

fn leaky_relu(x: &f64) -> f64 {
    if *x > 0.0 { *x } else { 0.01 * *x }
}

impl NeuralNetwork {
    pub fn new(nodes_values: Vec<usize>) -> Self {
        let mut nodes = Vec::new();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut deltas = Vec::new();

        for i in 0..nodes_values.len() {
            nodes.push(Matrix::new_and_fill(nodes_values[i], 1, 1.0).clone());
            if i > 0 {
                weights
                    .push(Matrix::new_and_fill(nodes_values[i], nodes_values[i - 1], 1.0).clone());
                biases.push(Matrix::new_and_fill(nodes_values[i], 1, 0.0));
                deltas.push(Matrix::new_and_fill(nodes_values[i], 1, 0.0));
            }
        }

        Self {
            nodes,
            weights,
            biases,
            deltas,
        }
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

            for j in 0..self.nodes[i].rows {
                self.nodes[i].set(j, 0, relu(&weighted_sum[j][0]))?;
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

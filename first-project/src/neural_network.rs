use crate::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    nodes: Vec<Matrix>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    deltas: Vec<Matrix>,
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
                weights.push(Matrix::new_and_fill(nodes_values[i], nodes_values[i - 1], 1.0).clone());
                biases.push(Matrix::new_and_fill(nodes_values[i], 1, 1.0));
                deltas.push(Matrix::new_and_fill(nodes_values[i], 1, 1.0));
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
            return Err("Input dimensions do not match the first layer of the network.".to_string());
        }

        self.nodes[0] = inputs.clone();

        for i in 1..self.nodes.len() {
            let weighted_sum = &self.weights[i - 1] * &self.nodes[i - 1] + &self.biases[i - 1];
        }

        Ok(())
    }
}

pub struct Matrix {
	pub rows: usize,
	pub cols: usize,
	pub data: [[f64; Row]; Col],
}

impl Matrix {
	pub fn new(rows: usize, cols: usize) -> Self {
		Self {
			rows,
			cols,
			data: [[0.0; usize]; usize],
		}
	}

	pub fn set_value(&mut self, row: usize, col: usize, value: f64) {
		self.data[row][col] = value;
	}

	pub fn get_value(&self, row: usize, col: usize) -> f64 {
		self.data[row][col]
	}
}
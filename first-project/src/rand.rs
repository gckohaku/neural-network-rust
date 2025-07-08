use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Rand {
	pub seed: u64,
}

// TODO: xorshift を使って乱数を生成する

impl Rand {
	pub fn new() -> Self {
		Self {
			seed: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
		}
	}

	pub fn new_with_seed(seed: u64) -> Self {
		Self { seed }
	}

	pub fn set_seed(&mut self, seed: u64) {
		self.seed = seed;
	}
}
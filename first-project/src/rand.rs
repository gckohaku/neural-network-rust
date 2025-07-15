use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Rand {
    x: u64,
    y: u64,
}

fn rotl(x: u64, k: u8) -> u64 {
    (&x << &k) | (&x >> (64 - &k))
}

// TODO: xoroshiro128** を使って乱数を生成する

impl Rand {
    pub fn new(&mut self) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Rand::new_with_seed(seed)
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            x: seed,
            y: rotl(rotl(seed, 32), 37),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.x = seed;
		self.y = rotl(rotl(seed, 32), 37);
    }

	// TODO: xoroshiro128** での乱数の生成
	pub fn next() {

	}
}

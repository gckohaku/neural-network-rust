use std::time;

use crate::{matrix::Matrix, rand::Rand};

pub fn calc() {
    let mut r = Rand::new_with_seed_u64(0x69DD7B40F36662E1);

    let mut m_50x50_a = Matrix::new(50, 50);
    let mut m_50x50_b = Matrix::new(50, 50);

    for i in 0..50 {
        for j in 0..50 {
            m_50x50_a.set(i, j, r.rand_f64()).unwrap();
            m_50x50_b.set(i, j, r.rand_f64()).unwrap();
        }
    }

    let mut m_100x100_a = Matrix::new(100, 100);
    let mut m_100x100_b = Matrix::new(100, 100);

    for i in 0..50 {
        for j in 0..50 {
            m_100x100_a.set(i, j, r.rand_f64()).unwrap();
            m_100x100_b.set(i, j, r.rand_f64()).unwrap();
        }
    }

    let mut m_200x200_a = Matrix::new(200, 200);
    let mut m_200x200_b = Matrix::new(200, 200);

    for i in 0..50 {
        for j in 0..50 {
            m_200x200_a.set(i, j, r.rand_f64()).unwrap();
            m_200x200_b.set(i, j, r.rand_f64()).unwrap();
        }
    }

    let mut m_500x500_a = Matrix::new(500, 500);
    let mut m_500x500_b = Matrix::new(500, 500);

	for i in 0 .. 50 {
		for j in 0..50 {
			m_500x500_a.set(i, j, r.rand_f64()).unwrap();
			m_500x500_b.set(i, j, r.rand_f64()).unwrap();
		}
	}

    let mut m_1000x1000_a = Matrix::new(1000, 1000);
    let mut m_1000x1000_b = Matrix::new(1000, 1000);

	for i in 0 .. 50 {
		for j in 0..50 {
			m_1000x1000_a.set(i, j, r.rand_f64()).unwrap();
			m_1000x1000_b.set(i, j, r.rand_f64()).unwrap();
		}
	}

	println!("start 50x50");
    let start_50x50 = time::Instant::now();
	let _ = &m_50x50_a * &m_50x50_b;
	println!("total calc time: {:?}\n", start_50x50.elapsed());

	println!("start 100x100");
    let start_100x100 = time::Instant::now();
	let _ = &m_100x100_a * &m_100x100_b;
	println!("total calc time: {:?}\n", start_100x100.elapsed());

	println!("start 200x200");
    let start_200x200 = time::Instant::now();
	let _ = &m_200x200_a * &m_200x200_b;
	println!("total calc time: {:?}\n", start_200x200.elapsed());

	println!("start 500x500");
    let start_500x500 = time::Instant::now();
	let _ = &m_500x500_a * &m_500x500_b;
	println!("total calc time: {:?}\n", start_500x500.elapsed());

	println!("start 1000x1000");
    let start_1000x1000 = time::Instant::now();
	let _ = &m_1000x1000_a * &m_1000x1000_b;
	println!("total calc time: {:?}\n", start_1000x1000.elapsed());
}

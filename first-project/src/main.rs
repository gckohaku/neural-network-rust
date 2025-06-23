use ndarray::*;
use ndarray_linalg::*;

fn main() {
    let a: Array2<i64> = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let b: Array2<i64> = arr2(&[[9, 8, 7], [6, 5, 4], [3, 2, 1]]);

    println!("resurt: {}", a.dot(&b));
}

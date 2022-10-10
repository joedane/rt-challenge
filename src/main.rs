
#![allow(dead_code)]

mod vec;
mod color;
mod canvas;

use vec::Vector;

fn main() {
    let v: Vector<f64> = Vector::new(1., 2., 3.);
    let v1 = Vector::new(3., 3., 3.);
    println!("v = {:?}", v + v1);
}


#![allow(dead_code)]

mod vec;
mod color;
mod canvas;
mod shape;


use vec::{Vector, Point, Matrix};
use shape::{Object};

use num_traits::float::Float;


fn main() {
    let v: Vector<f64> = Vector::new(1., 2., 3.);
    let v1 = Vector::new(3., 3., 3.);
    println!("v = {:?}", v + v1);
}


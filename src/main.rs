
#![feature(generic_const_exprs)]

mod vec;
mod color;
mod canvas;

use vec::{Vector, Point};

fn main() {
    let v = Vector::new(1., 2., 3.);
    let v1 = Vector::new(3., 3., 3.);
    println!("v = {:?}", v + v1);
}

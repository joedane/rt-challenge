#![allow(dead_code)]

mod canvas;
mod color;
mod shape;
mod vec;

use shape::{Material, Object};
use vec::{Matrix, Point, Vector};
use color::Color;

use num_traits::float::Float;

struct Light<T: Float> {
    position: Point<T>,
    intensity: Color,
}

fn phong<T: Float>(
    m: Material<T>,
    l: Light<T>,
    p: Point<T>,
    eyev: Vector<T>,
    normal: Vector<T>,
) -> Color {
    let effective_color = m.color * l.intensity;
    let light_v = (l.position - p).normalize();
    let ambient = effective_color * m.ambient;
    let light_dot_normal = Vector::dot(light_v, normal);
    if light_dot_normal < T::zero() {
        ambient
    } else {
        let diffuse = effective_color * m.diffuse * light_dot_normal;
        let reflect_v = -light_v.reflect(&normal);
        let reflect_dot_eye = Vector::dot(reflect_v, eyev);
        
        if reflect_dot_eye < T::zero() {
            ambient + diffuse
        } else {
            let factor = reflect_dot_eye.powf(m.shininess);
            ambient + diffuse + l.intensity * m.specular * factor
        }
    }
}

fn main() {
    let v: Vector<f64> = Vector::new(1., 2., 3.);
    let v1 = Vector::new(3., 3., 3.);
    println!("v = {:?}", v + v1);
}

#![allow(dead_code)]

mod canvas;
mod color;
mod shape;
mod vec;

use shape::{Material, Object, Sphere, Ray, Intersections, Intersection, Hit};
use vec::{Matrix, Point, Vector, TransformBuilder};
use color::Color;

use num_traits::float::Float;

struct Light<T: Float> {
    position: Point<T>,
    intensity: Color,
}

impl<T:Float> Light<T> {

    fn new(position: Point<T>, intensity: Color) -> Self {
        Light { position, intensity }
    }
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

struct World<'a> {
    objects: Vec<Object<'a, f64>>,
    lights: Vec<Light<f64>>,
}

impl<'a> Default for World<'a> {
    fn default() -> Self {
        let mut o = vec![];
        let s = Sphere::<f64>::default();
        let mut m: Material<f64> = Material::default();
        m.color = Color::new(0.8, 1.0, 0.6);
        m.diffuse = 0.7;
        m.specular = 0.2;
        o.push(Object::<f64>::new(Sphere::new(), m));
        
        let mut s = Sphere::<f64>::default();
        s.set_transform(TransformBuilder::new().scaling(0.5, 0.5, 0.5).finish());
        o.push(Object::<f64>::new(s, Material::default()));

        World { objects: o, lights: vec![Light::new(Point::new(-10, 10, -10), Color::new(1., 1., 1.))]}
    }
}

impl<'a> World<'a> {

    fn intersect(&self, ray: &Ray<f64>) -> Intersections<f64> {
        let mut is = Intersections::new();
        for o in self.objects.iter() {
            match o.shape.intersect(ray) {
                Hit::One(h) => {
                    is.add(Intersection::new(h, o.shape.as_ref()))
                },
                Hit::Two(h1, h2) => {
                    is.add(Intersection::new(h1, o.shape.as_ref()));
                    is.add(Intersection::new(h2, o.shape.as_ref()));
                }
                _ => {}
            }
        }
        is
    }
}
fn main() {
    let v: Vector<f64> = Vector::new(1., 2., 3.);
    let v1 = Vector::new(3., 3., 3.);
    println!("v = {:?}", v + v1);
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use crate::Light;
    use crate::vec::{Vector, Point};
    use crate::color::Color;
    use crate::shape::Material;
    use super::*;

    #[test]
    fn test_lighting() {
        let material = Material::default();
        let position: Point<f64> = Point::new(0, 0, 0);
        let light: Light<f64> = Light::new(Point::new(0, 0, -10), Color::new(1u8, 1u8, 1u8));
        let eyev: Vector<f64> = Vector::new(0, 0, -1);
        let normal: Vector<f64> = Vector::new(0, 0, -1);    
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 1.9, epsilon = 0.0001);
        assert_relative_eq!(r.green, 1.9, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 1.9, epsilon = 0.0001);

        let s2 = 2.0.sqrt() / 2.0;
        let material = Material::default();
        let position: Point<f64> = Point::new(0, 0, 0);
        let light: Light<f64> = Light::new(Point::new(0, 0, -10), Color::new(1u8, 1u8, 1u8));
        let eyev: Vector<f64> = Vector::new(0., s2, -s2);
        let normal: Vector<f64> = Vector::new(0, 0, -1);    
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 1.0, epsilon = 0.0001);
        assert_relative_eq!(r.green, 1.0, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 1.0, epsilon = 0.0001);

        let material = Material::default();
        let position: Point<f64> = Point::new(0, 0, 0);
        let light: Light<f64> = Light::new(Point::new(0, 10, -10), Color::new(1u8, 1u8, 1u8));
        let eyev: Vector<f64> = Vector::new(0, 0, -1);
        let normal: Vector<f64> = Vector::new(0, 0, -1);    
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 0.7364, epsilon = 0.0001);
        assert_relative_eq!(r.green, 0.7364, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 0.7364, epsilon = 0.0001);
        
        let material = Material::default();
        let position: Point<f64> = Point::new(0, 0, 0);
        let light: Light<f64> = Light::new(Point::new(0, 10, -10), Color::new(1u8, 1u8, 1u8));
        let eyev: Vector<f64> = Vector::new(0., -s2, -s2);
        let normal: Vector<f64> = Vector::new(0, 0, -1);    
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 1.6364, epsilon = 0.0001);
        assert_relative_eq!(r.green, 1.6364, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 1.6364, epsilon = 0.0001);

        let material = Material::default();
        let position: Point<f64> = Point::new(0, 0, 0);
        let light: Light<f64> = Light::new(Point::new(0, 0, 10), Color::new(1u8, 1u8, 1u8));
        let eyev: Vector<f64> = Vector::new(0, 0, -1);
        let normal: Vector<f64> = Vector::new(0, 0, -1);    
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 0.1, epsilon = 0.0001);
        assert_relative_eq!(r.green, 0.1, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 0.1, epsilon = 0.0001);


    }

    #[test]
    fn test_default_world() {
        let w = World::default();
        let r: Ray<f64> = Ray::new(Point::new(0, 0, -5), Vector::new(0, 0, 1));
        let is = w.intersect(&r);
        println!("FOO: {:#?}", is.get_ts());
        assert_eq!(is.len(), 4);
        assert!(is.at_p(4.));
        assert!(is.at_p(4.5));
        assert!(is.at_p(5.5));
        assert!(is.at_p(6.));
    }
}

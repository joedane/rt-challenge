use crate::color::Color;
use crate::vec::{Matrix, Point, Vector};
use crate::MyFloat2;

use std::cmp::{Eq, Ord, PartialEq, PartialOrd, Reverse};
use std::collections::BinaryHeap;
use std::fmt::Write;

pub struct Ray {
    pub o: Point,
    pub d: Vector,
}

impl Ray {
    pub fn new(o: Point, d: Vector) -> Self {
        Ray { o, d }
    }

    pub fn position<P: Into<MyFloat2>>(&self, t: P) -> Point {
        self.o + self.d * t.into()
    }

    pub fn transform(&self, xf: &Matrix) -> Ray {
        Ray::new(xf * self.o, xf * self.d)
    }
}

pub trait Shape {
    fn local_intersect(&self, r: &Ray) -> Hit;

    fn local_normal(&self, at: Point) -> Vector;

    fn get_transform(&self) -> &Matrix;

    fn describe(&self) -> String;

    fn intersect(&self, r: &Ray) -> Hit {
        // optimize?
        let ixf = self.get_transform().invert();
        let local_ray = r.transform(&ixf);
        self.local_intersect(&local_ray)
    }

    fn normal(&self, at: Point) -> Vector {
        let xf = self.get_transform().invert();
        let local_point = &xf * at;
        let local_normal = self.local_normal(local_point);
        let world_normal = &(xf.transpose()) * local_normal;
        world_normal.normalize()
    }
}

#[derive(Clone, Copy)]
pub struct Material {
    pub color: Color,
    pub ambient: MyFloat2,
    pub diffuse: MyFloat2,
    pub specular: MyFloat2,
    pub shininess: MyFloat2,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            color: Color::new(1., 1., 1.),
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.0,
        }
    }
}

pub struct Object<'a> {
    pub shape: Box<dyn Shape + 'a>,
    pub material: Material,
}

impl<'a> Object<'a> {
    pub fn new<S: Shape + 'a>(s: S, m: Material) -> Self {
        Object {
            shape: Box::new(s),
            material: m,
        }
    }

    pub fn describe_into(&self, s: &mut String) {
        write!(s, "Object of type {}", self.shape.describe()).unwrap();
    }
}
pub struct Intersection<'a> {
    pub pos: MyFloat2,
    pub shape: &'a dyn Shape,
    pub material: Option<Material>,
}

impl<'a> Intersection<'a> {
    pub fn new<P: Into<MyFloat2>>(pos: P, shape: &'a dyn Shape) -> Self {
        Intersection {
            pos: pos.into(),
            shape,
            material: None,
        }
    }

    pub fn new_with_material<P: Into<MyFloat2>>(
        pos: P,
        shape: &'a dyn Shape,
        material: Material,
    ) -> Self {
        Intersection {
            pos: pos.into(),
            shape,
            material: Some(material),
        }
    }
}

impl<'a> Ord for Intersection<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.pos.partial_cmp(&other.pos) {
            None => {
                panic!()
            }
            Some(o) => o,
        }
    }
}

impl<'a> PartialOrd for Intersection<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> PartialEq for Intersection<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
    }
}

impl<'a> Eq for Intersection<'a> {}
pub struct Intersections<'a> {
    storage: BinaryHeap<Reverse<Intersection<'a>>>,
}

impl<'a> Intersections<'a> {
    pub fn new() -> Self {
        Self {
            storage: BinaryHeap::new(),
        }
    }

    pub fn add(&mut self, i: Intersection<'a>) {
        self.storage.push(Reverse(i));
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn get_hit(&self) -> Option<&Intersection<'a>> {
        if self.storage.is_empty() {
            None
        } else {
            let r = self.storage.iter().find(|&r| r.0.pos > 0.0);
            match r {
                None => None,
                Some(r) => Some(&r.0),
            }
        }
    }

    pub fn at_p(&self, p: MyFloat2) -> bool {
        self.storage.iter().any(|item| item.0.pos == p)
    }

    pub fn get_ts(&self) -> Vec<MyFloat2> {
        self.storage.iter().map(|item| item.0.pos).collect()
    }
}

pub enum Hit {
    None,
    One(MyFloat2),
    Two(MyFloat2, MyFloat2),
}

impl Hit {
    pub fn is_some(&self) -> bool {
        matches!(self, Hit::None)
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }
    fn at_p<P: Into<MyFloat2>>(&self, p: P) -> bool {
        let p = p.into();
        match self {
            Hit::None => false,
            Hit::One(this_p) => *this_p == p,
            Hit::Two(p1, p2) => *p1 == p || *p2 == p,
        }
    }
}

impl std::fmt::Debug for Hit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::One(arg0) => f.debug_tuple("One").field(arg0).finish(),
            Self::Two(arg0, arg1) => f.debug_tuple("Two").field(arg0).field(arg1).finish(),
        }
    }
}

pub struct Sphere {
    xf: Matrix,
}

impl Sphere {
    pub fn new() -> Self {
        Sphere {
            xf: Matrix::identity(),
        }
    }

    pub fn set_transform(&mut self, xf: Matrix) {
        self.xf = xf;
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Sphere::new()
    }
}

impl Shape for Sphere {
    fn describe(&self) -> String {
        let s = format!("Sphere {{ origin: {:?} }}", &(self.xf) * Point::at_origin());
        s
    }

    fn get_transform(&self) -> &Matrix {
        &self.xf
    }

    fn local_intersect(&self, r: &Ray) -> Hit {
        let s = r.o - Point::at_origin();
        let a = Vector::dot(r.d, r.d);
        let b = Vector::dot(r.d, s);
        let b = b + b;
        let c = Vector::dot(s, s) - 1.0;
        let discrim = b * b - 4.0 * a * c;
        if discrim < 0.0 {
            Hit::None
        } else {
            let t1 = (-b - discrim.sqrt()) / (2.0 * a);
            let t2 = (-b + discrim.sqrt()) / (2.0 * a);
            if t1 == t2 {
                Hit::One(t1)
            } else {
                Hit::Two(t1, t2)
            }
        }
    }

    fn local_normal(&self, at: Point) -> Vector {
        (at - Point::new(0.0, 0.0, 0.0)).normalize()
    }
}
#[cfg(test)]
mod test {

    use approx::assert_relative_eq;

    use super::*;
    use crate::vec::TransformBuilder;

    #[test]
    fn test_ray_position() {
        let r: Ray = Ray::new(Point::new(2, 3, 4), Vector::new(1, 0, 0));
        assert_eq!(r.position(0), Point::new(2, 3, 4));
        assert_eq!(r.position(1), Point::new(3, 3, 4));
        assert_eq!(r.position(-1), Point::new(1, 3, 4));
        assert_eq!(r.position(2.5), Point::new(4.5, 3., 4.));
    }

    #[test]
    fn test_intersect() {
        let r: Ray = Ray::new(Point::new(0, 0, -5), Vector::new(0, 0, 1));
        let s = Sphere::new();
        let i = s.intersect(&r);
        assert!(i.is_some());
        assert!(i.at_p(4));
        assert!(i.at_p(6));

        let r: Ray = Ray::new(Point::new(0, 1, -5), Vector::new(0, 0, 1));
        let i = s.intersect(&r);
        assert!(i.is_some());
        assert!(matches!(i, Hit::One(_)));
        assert!(i.at_p(5));

        let r: Ray = Ray::new(Point::new(0, 2, -5), Vector::new(0, 0, 1));
        let i = s.intersect(&r);
        assert!(i.is_none());

        let r: Ray = Ray::new(Point::new(0, 0, 0), Vector::new(0, 0, 1));
        let i = s.intersect(&r);
        assert!(i.is_some());
        assert!(i.at_p(-1));
        assert!(i.at_p(1));

        let r: Ray = Ray::new(Point::new(0, 0, 5), Vector::new(0, 0, 1));
        let i = s.intersect(&r);
        assert!(i.is_some());
        assert!(i.at_p(-6));
        assert!(i.at_p(-4));
    }

    #[test]
    fn test_intersections() {
        let mut is: Intersections<'_> = Intersections::new();
        let s = Sphere::new();
        is.add(Intersection::new(1, &s));
        is.add(Intersection::new(2, &s));
        assert_eq!(is.get_hit().unwrap().pos, 1.0);

        let mut is: Intersections<'_> = Intersections::new();
        is.add(Intersection::new(-2, &s));
        is.add(Intersection::new(-1, &s));
        assert!(is.get_hit().is_none());

        let mut is: Intersections<'_> = Intersections::new();
        is.add(Intersection::new(-2, &s));
        is.add(Intersection::new(1, &s));
        assert_eq!(is.get_hit().unwrap().pos, 1.0);

        let mut is: Intersections<'_> = Intersections::new();
        is.add(Intersection::new(5, &s));
        is.add(Intersection::new(7, &s));
        is.add(Intersection::new(-3, &s));
        is.add(Intersection::new(2, &s));
        assert_eq!(is.get_hit().unwrap().pos, 2.0);
    }

    #[test]
    fn test_ray_xf() {
        let r: Ray = Ray::new(Point::new(1, 2, 3), Vector::new(0, 1, 0));
        let xf = TransformBuilder::new().translate(3, 4, 5).finish();
        let r2 = r.transform(&xf);
        assert_eq!(r2.o, Point::new(4, 6, 8));
        assert_eq!(r2.d, Vector::new(0, 1, 0));

        let xf = TransformBuilder::new().scaling(2, 3, 4).finish();
        let r2 = r.transform(&xf);
        assert_eq!(r2.o, Point::new(2, 6, 12));
        assert_eq!(r2.d, Vector::new(0, 3, 0));
    }

    #[test]
    fn test_sphere_intersect() {
        let mut s = Sphere::default();
        let ray: Ray = Ray::new(Point::new(0, 0, -5), Vector::new(0, 0, 1));
        s.set_transform(TransformBuilder::new().scaling(2, 2, 2).finish());
        if let Hit::Two(h1, h2) = s.intersect(&ray) {
            assert!((h1 == 3.0 || h1 == 7.0) && (h2 == 3.0 || h2 == 7.0));
        } else {
            panic!();
        }
    }

    #[test]
    fn test_sphere_normal() {
        use std::f64::consts::FRAC_1_SQRT_2;

        let mut s = Sphere::new();

        let n = s.normal(Point::new(1, 0, 0));
        assert_eq!(n, Vector::new(1, 0, 0));

        let n = s.normal(Point::new(0, 1, 0));
        assert_eq!(n, Vector::new(0, 1, 0));

        let n = s.normal(Point::new(0, 0, 1));
        assert_eq!(n, Vector::new(0, 0, 1));

        let r3 = 3.0f32.sqrt() / 3.0;
        let n = s.normal(Point::new(r3, r3, r3));
        assert_relative_eq!(n, Vector::new(r3, r3, r3), epsilon = 0.000001);

        s.set_transform(TransformBuilder::new().translate(0, 1, 0).finish());
        let n = s.normal(Point::new(0., 1.70711, -FRAC_1_SQRT_2));

        assert_relative_eq!(
            n,
            Vector::new(0., FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            epsilon = 0.00001
        );

        s.set_transform(
            TransformBuilder::new()
                .rotation_z(std::f64::consts::PI / 5.0)
                .scaling(1., 0.5, 1.)
                .finish(),
        );
        let r2 = 2.0f32.sqrt() / 2.0;

        let n = s.normal(Point::new(0., r2, -r2));
        assert_relative_eq!(n, Vector::new(0., 0.97014, -0.24254), epsilon = 0.00001);
    }

    #[test]
    fn test_camera2() {}
}

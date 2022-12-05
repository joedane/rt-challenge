#![allow(dead_code)]

mod camera;
mod canvas;
mod color;
mod shape;
mod vec;

use color::Color;
use shape::{Hit, Intersection, Intersections, Material, Object, Ray, Sphere};
use vec::{Point, TransformBuilder, Vector};

//use num_traits::float::Float;
use tracing::Level;

type MyFloat2 = f64;

const TOLERANCE: MyFloat2 = 0.00001;

#[derive(Clone, Copy)]
struct Light {
    position: Point,
    intensity: Color,
}

impl Light {
    fn new(position: Point, intensity: Color) -> Self {
        Light {
            position,
            intensity,
        }
    }
}

impl Default for Light {
    fn default() -> Self {
        Self {
            position: Point::new(0, 0, -10),
            intensity: Color::WHITE,
        }
    }
}
fn phong(m: Material, l: Light, p: Point, eyev: Vector, normal: Vector) -> Color {
    let effective_color = m.color * l.intensity;
    let light_v = (l.position - p).normalize();
    let ambient = effective_color * m.ambient;
    let light_dot_normal = Vector::dot(light_v, normal);
    if light_dot_normal < 0.0 {
        ambient
    } else {
        let diffuse = effective_color * m.diffuse * light_dot_normal;
        let reflect_v = -light_v.reflect(&normal);
        let reflect_dot_eye = Vector::dot(reflect_v, eyev);

        if reflect_dot_eye < 0.0 {
            ambient + diffuse
        } else {
            let factor = reflect_dot_eye.powf(m.shininess);
            ambient + diffuse + l.intensity * m.specular * factor
        }
    }
}

struct World<'a> {
    objects: Vec<Object<'a>>,
    light: Light,
}

impl<'a> Default for World<'a> {
    fn default() -> Self {
        let mut o: Vec<Object> = vec![];
        let s = Sphere::default();
        let m: Material = shape::Material {
            color: Color::new(0.8, 1.0, 0.6),
            diffuse: 0.7,
            specular: 0.2,
            ..Default::default()
        };
        o.push(Object::new(s, m));

        let mut s = Sphere::default();
        s.set_transform(TransformBuilder::new().scaling(0.5, 0.5, 0.5).finish());
        o.push(Object::new(s, Material::default()));

        World {
            objects: o,
            light: Light::new(Point::new(-10, 10, -10), Color::new(1., 1., 1.)),
        }
    }
}

impl<'a> World<'a> {
    pub(crate) fn new(objects: Vec<Object<'a>>, light: Light) -> Self {
        Self { objects, light }
    }

    fn intersect(&self, ray: &Ray) -> Intersections {
        let mut is = Intersections::new();
        for o in self.objects.iter() {
            match o.shape.intersect(ray) {
                Hit::One(h) => is.add(Intersection::new_with_material(
                    h,
                    o.shape.as_ref(),
                    o.material,
                )),
                Hit::Two(h1, h2) => {
                    is.add(Intersection::new_with_material(
                        h1,
                        o.shape.as_ref(),
                        o.material,
                    ));
                    is.add(Intersection::new_with_material(
                        h2,
                        o.shape.as_ref(),
                        o.material,
                    ));
                }
                _ => {}
            }
        }
        is
    }

    fn get_obj(&self, i: usize) -> &Object<'a> {
        self.objects.get(i).unwrap()
    }

    fn get_obj_mut(&mut self, i: usize) -> &mut Object<'a> {
        self.objects.get_mut(i).unwrap()
    }

    pub fn describe(&self) -> String {
        let mut s = "World containing: ".to_string();
        for obj in &self.objects {
            obj.describe_into(&mut s);
            s.push('\n');
        }
        s
    }
}

#[derive(Clone, Copy)]
struct HitComputations {
    t: MyFloat2,
    material: Option<Material>,
    point: Point,
    eyev: Vector,
    normal: Vector,
    inside: bool,
}

fn prepare_computations(i: &Intersection, ray: &Ray) -> HitComputations {
    let point = ray.position(i.pos);
    let eyev = -ray.d;
    let mut normal = i.shape.normal(point);
    let inside;
    if Vector::dot(normal, eyev) < 0.0 {
        inside = true;
        normal = -normal;
    } else {
        inside = false;
    }
    HitComputations {
        t: i.pos,
        material: i.material,
        point,
        eyev,
        normal,
        inside,
    }
}

fn shade_hit(w: &World, comps: HitComputations) -> Color {
    phong(
        comps.material.unwrap(),
        w.light,
        comps.point,
        comps.eyev,
        comps.normal,
    )
}

fn color_at<'a>(w: &'a World<'a>, ray: Ray) -> Color {
    let is = w.intersect(&ray);
    match is.get_hit() {
        None => Color::new(0., 0., 0.),
        Some(h) => shade_hit(w, prepare_computations(h, &ray)),
    }
}

fn main() {
    let tracing = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();
    tracing::subscriber::set_global_default(tracing).expect("failed to set tracing subscriber");

    let world = World::default();
    let camera = camera::Camera::default();
    let mut canvas: canvas::Canvas<400, 400> = canvas::Canvas::new();
    camera.render(&world, &mut canvas);
    canvas.write("img.jpg");
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use super::*;
    use crate::color::Color;
    use crate::shape::*;
    use crate::vec::{Point, Vector};
    use crate::Light;

    #[test]
    fn test_lighting() {
        let material = Material::default();
        let position: Point = Point::new(0, 0, 0);
        let light: Light = Light::new(Point::new(0, 0, -10), Color::new(1u8, 1u8, 1u8));
        let eyev: Vector = Vector::new(0, 0, -1);
        let normal: Vector = Vector::new(0, 0, -1);
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 1.9, epsilon = 0.0001);
        assert_relative_eq!(r.green, 1.9, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 1.9, epsilon = 0.0001);

        let s2 = 2.0f64.sqrt() / 2.0;
        let material = Material::default();
        let position = Point::new(0, 0, 0);
        let light = Light::new(Point::new(0, 0, -10), Color::new(1u8, 1u8, 1u8));
        let eyev = Vector::new(0., s2, -s2);
        let normal = Vector::new(0, 0, -1);
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 1.0, epsilon = 0.0001);
        assert_relative_eq!(r.green, 1.0, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 1.0, epsilon = 0.0001);

        let material = Material::default();
        let position = Point::new(0, 0, 0);
        let light = Light::new(Point::new(0, 10, -10), Color::new(1u8, 1u8, 1u8));
        let eyev = Vector::new(0, 0, -1);
        let normal = Vector::new(0, 0, -1);
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 0.7364, epsilon = 0.0001);
        assert_relative_eq!(r.green, 0.7364, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 0.7364, epsilon = 0.0001);

        let material = Material::default();
        let position = Point::new(0, 0, 0);
        let light = Light::new(Point::new(0, 10, -10), Color::new(1u8, 1u8, 1u8));
        let eyev = Vector::new(0., -s2, -s2);
        let normal = Vector::new(0, 0, -1);
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 1.6364, epsilon = 0.0001);
        assert_relative_eq!(r.green, 1.6364, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 1.6364, epsilon = 0.0001);

        let material = Material::default();
        let position = Point::new(0, 0, 0);
        let light = Light::new(Point::new(0, 0, 10), Color::new(1u8, 1u8, 1u8));
        let eyev = Vector::new(0, 0, -1);
        let normal = Vector::new(0, 0, -1);
        let r = phong(material, light, position, eyev, normal);
        assert_relative_eq!(r.red, 0.1, epsilon = 0.0001);
        assert_relative_eq!(r.green, 0.1, epsilon = 0.0001);
        assert_relative_eq!(r.blue, 0.1, epsilon = 0.0001);
    }

    #[test]
    fn test_default_world() {
        let w = World::default();
        let r: Ray = Ray::new(Point::new(0, 0, -5), Vector::new(0, 0, 1));
        let is = w.intersect(&r);
        assert_eq!(is.len(), 4);
        assert!(is.at_p(4.));
        assert!(is.at_p(4.5));
        assert!(is.at_p(5.5));
        assert!(is.at_p(6.));
    }

    #[test]
    fn test_comps() {
        let ray: Ray = Ray::new(Point::new(0, 0, -5), Vector::new(0, 0, 1));
        let s = Sphere::default();
        let i = Intersection::new(4, &s);
        let comps = prepare_computations(&i, &ray);
        assert!(!comps.inside);

        let ray: Ray = Ray::new(Point::new(0, 0, 0), Vector::new(0, 0, 1));
        let i = Intersection::new(1, &s);
        let comps = prepare_computations(&i, &ray);
        assert!(comps.inside);
        assert_eq!(comps.point, Point::new(0, 0, 1));
        assert_eq!(comps.eyev, Vector::new(0, 0, -1));
        assert_eq!(comps.normal, Vector::new(0, 0, -1));
    }

    #[test]
    fn test_color_world() {
        let w: World = World::default();
        let ray = Ray::new(Point::new(0, 0, -5), Vector::new(0, 0, 1));
        let o = w.get_obj(0);
        let i = Intersection::new_with_material(4, o.shape.as_ref(), o.material);
        let comps = prepare_computations(&i, &ray);
        let c = shade_hit(&w, comps);
        assert_relative_eq!(c.red, 0.38066, epsilon = 0.00001);
        assert_relative_eq!(c.green, 0.47583, epsilon = 0.00001);
        assert_relative_eq!(c.blue, 0.2855, epsilon = 0.00001);

        let w = World {
            light: Light::new(Point::new(0., 0.25, 0.), Color::new(1., 1., 1.)),
            ..Default::default()
        };
        let ray = Ray::new(Point::new(0, 0, 0), Vector::new(0, 0, 1));
        let o = w.get_obj(1);
        let i = Intersection::new_with_material(0.5, o.shape.as_ref(), o.material);
        let comps = prepare_computations(&i, &ray);
        let c = shade_hit(&w, comps);
        assert_relative_eq!(c.red, 0.90498, epsilon = 0.00001);
        assert_relative_eq!(c.green, 0.90498, epsilon = 0.00001);
        assert_relative_eq!(c.blue, 0.90498, epsilon = 0.00001);

        let w = World::default();
        let ray = Ray::new(Point::new(0, 0, -5), Vector::new(0, 1, 0));
        let c = color_at(&w, ray);
        assert_eq!(c, Color::new(0., 0., 0.));

        let ray = Ray::new(Point::new(0, 0, -5), Vector::new(0, 0, 1));
        let c = color_at(&w, ray);
        assert_relative_eq!(c.red, 0.38066, epsilon = 0.00001);
        assert_relative_eq!(c.green, 0.47583, epsilon = 0.00001);
        assert_relative_eq!(c.blue, 0.2855, epsilon = 0.00001);

        let mut w = World::default();
        w.get_obj_mut(0).material.ambient = 1.0;
        w.get_obj_mut(1).material.ambient = 1.0;
        let ray = Ray::new(Point::new(0., 0., 0.75), Vector::new(0, 0, -1));
        let c = color_at(&w, ray);
        assert_eq!(c, w.get_obj(1).material.color);
    }
}

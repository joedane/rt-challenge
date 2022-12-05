use crate::canvas::Canvas;
use crate::color_at;
use crate::vec::{Matrix, Vector};
use crate::Point;
use crate::Ray;
use crate::World;

pub struct Camera {
    hsize: usize,
    vsize: usize,
    half_width: f64,
    half_height: f64,
    fov: f64,
    pixel_size: f64,
    xf: Matrix,
}

impl Default for Camera {
    fn default() -> Self {
        Camera::new_with_transform(
            400,
            400,
            std::f64::consts::PI / 4.0,
            Point::new(0.0f32, 0.0f32, -5.0f32),
            Point::new(0, 0, 0),
            Vector::new(0, 1, 0),
        )
    }
}

impl Camera {
    fn new(hsize: usize, vsize: usize, fov: f64) -> Self {
        let half_view: f64 = (fov / 2.0f64).tan();
        let aspect: f64 = hsize as f64 / vsize as f64;
        let (hw, hh);
        if aspect > 1.0 {
            hw = half_view;
            hh = half_view / aspect;
        } else {
            hw = half_view * aspect;
            hh = half_view
        }
        let pixel_size: f64 = (hw * 2.0) / hsize as f64;
        Camera {
            hsize,
            vsize,
            half_width: hw,
            half_height: hh,
            fov,
            pixel_size,
            xf: Matrix::identity(),
        }
    }

    fn new_with_transform(
        hsize: usize,
        vsize: usize,
        fov: f64,
        from: Point,
        to: Point,
        up: Vector,
    ) -> Self {
        let mut s = Self::new(hsize, vsize, fov);
        s.set_transform(Self::make_view_transform(from, to, up));
        s
    }

    fn make_view_transform(from: Point, to: Point, up: Vector) -> Matrix {
        let forward = Vector::normalize(&(to - from));
        let up = Vector::normalize(&up);
        let left = Vector::cross(forward, up);
        let up = Vector::cross(left, forward);

        let o = Matrix::new(vec![
            left.x, left.y, left.z, 0.0, up.x, up.y, up.z, 0.0, -forward.x, -forward.y, -forward.z,
            0.0, 0.0, 0.0, 0.0, 1.0,
        ]);

        &o * &Matrix::make_translation(-from.x, -from.y, -from.z)
    }

    pub(crate) fn set_transform(&mut self, xf: Matrix) {
        self.xf = xf;
    }

    fn ray_for_pixel(&self, x: usize, y: usize) -> Ray {
        let xoffset = (x as f64 + 0.5) * self.pixel_size;
        let yoffset = (y as f64 + 0.5) * self.pixel_size;
        let world_x = self.half_width - xoffset;
        let world_y = self.half_height - yoffset;
        let pixel = &self.xf.invert() * Point::new(world_x, world_y, -1.0);
        let origin = &self.xf.invert() * Point::new(0.0, 0.0, 0.0);

        Ray::new(origin, (pixel - origin).normalize())
    }

    pub(crate) fn render<const W: usize, const H: usize>(
        &self,
        world: &World,
        canvas: &mut Canvas<W, H>,
    ) {
        for y in 0..self.vsize {
            for x in 0..self.hsize {
                // debug!(x, y);
                let r = self.ray_for_pixel(x, y);
                canvas.set(x, y, color_at(world, r));
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        shape::{Material, Object, Sphere},
        vec::*,
        Light,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_camera() {
        let c: Camera = Camera::new(200, 125, std::f64::consts::PI / 2.0);
        assert_relative_eq!(c.pixel_size, 0.01, epsilon = 0.001);

        let c: Camera = Camera::new(125, 200, std::f64::consts::PI / 2.0);
        assert_relative_eq!(c.pixel_size, 0.01, epsilon = 0.001);

        let c: Camera = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let r = c.ray_for_pixel(100, 50);
        assert_relative_eq!(r.o, Point::new(0, 0, 0), epsilon = 0.0001);
        assert_relative_eq!(r.d, Vector::new(0, 0, -1), epsilon = 0.0001);

        let r = c.ray_for_pixel(0, 0);
        assert_relative_eq!(r.o, Point::new(0, 0, 0));
        assert_relative_eq!(
            r.d,
            Vector::new(0.66519, 0.33259, -0.66851),
            epsilon = 0.00001
        );

        let mut c: Camera = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let s2 = 2.0f64.sqrt() / 2.0;
        c.xf = TransformBuilder::new()
            .translate(0, -2, 5)
            .rotation_y(std::f64::consts::PI / 4.0)
            .finish();
        let r = c.ray_for_pixel(100, 50);
        assert_relative_eq!(r.o, Point::new(0, 2, -5));
        assert_relative_eq!(r.d, Vector::new(s2, 0., -s2), epsilon = 0.00001);

        let xf = Camera::make_view_transform(
            Point::new(0, 0, 0),
            Point::new(0, 0, -1),
            Vector::new(0, 1, 0),
        );
        assert_relative_eq!(xf, Matrix::identity());

        let xf = Camera::make_view_transform(
            Point::new(0, 0, 0),
            Point::new(0, 0, 1),
            Vector::new(0, 1, 0),
        );
        assert_relative_eq!(xf, Matrix::make_scaling(-1, 1, -1));

        let xf = Camera::make_view_transform(
            Point::new(0, 0, 8),
            Point::new(0, 0, 0),
            Vector::new(0, 1, 0),
        );
        assert_relative_eq!(xf, Matrix::make_translation(0, 0, -8));

        let xf = Camera::make_view_transform(
            Point::new(1, 3, 2),
            Point::new(4, -2, 8),
            Vector::new(1, 1, 0),
        );
        assert_relative_eq!(
            xf,
            Matrix::new(vec![
                -0.50709, 0.50709, 0.67612, -2.36643, 0.76772, 0.60609, 0.12122, -2.82843,
                -0.35857, 0.59761, -0.71714, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000,
            ]),
            epsilon = 0.00001
        );
    }

    #[test]
    fn test_camera2() {
        let _c = Camera::new_with_transform(
            400,
            400,
            1.5,
            Point::new(0, 0, -3),
            Point::new(0, 0, 0),
            Vector::new(0, 1, 0),
        );

        let w: World = World::default();
        println!("{}", w.describe());
    }

    #[test]
    fn test_camera3() {
        let c = Camera::default();
        let _s = Sphere::default();
        let r = c.ray_for_pixel(c.hsize / 2, c.vsize / 2);
        let w = World::new(
            vec![Object::new(Sphere::default(), Material::default())],
            Light::default(),
        );

        println!("color: {:?}", color_at(&w, r));
    }
}

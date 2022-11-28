use crate::canvas::Canvas;
use crate::vec::{Matrix, Vector};
use crate::Point;
use crate::Ray;
use crate::World;
use crate::{color_at, MyFloat};
use tracing::debug;

pub struct Camera<T> {
    hsize: usize,
    vsize: usize,
    half_width: f64,
    half_height: f64,
    fov: f64,
    pixel_size: f64,
    xf: Matrix<T>,
}

impl<T: MyFloat> Default for Camera<T> {
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

impl<T: MyFloat> Camera<T> {
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
        from: Point<T>,
        to: Point<T>,
        up: Vector<T>,
    ) -> Self {
        let mut s = Self::new(hsize, vsize, fov);
        s.set_transform(Self::make_view_transform(from, to, up));
        s
    }

    fn make_view_transform(from: Point<T>, to: Point<T>, up: Vector<T>) -> Matrix<T> {
        let forward = Vector::normalize(&(to - from));
        let up = Vector::normalize(&up);
        let left = Vector::cross(forward, up);
        let up = Vector::cross(left, forward);

        let o = Matrix::new(vec![
            left.x,
            left.y,
            left.z,
            T::zero(),
            up.x,
            up.y,
            up.z,
            T::zero(),
            -forward.x,
            -forward.y,
            -forward.z,
            T::zero(),
            T::zero(),
            T::zero(),
            T::zero(),
            T::one(),
        ]);

        &o * &Matrix::make_translation(-from.x, -from.y, -from.z)
    }

    pub(crate) fn set_transform(&mut self, xf: Matrix<T>) {
        self.xf = xf;
    }

    fn ray_for_pixel(&self, x: usize, y: usize) -> Ray<T> {
        let xoffset = (x as f64 + 0.5) * self.pixel_size;
        let yoffset = (y as f64 + 0.5) * self.pixel_size;
        let world_x = self.half_width - xoffset;
        let world_y = self.half_height - yoffset;
        let pixel = &self.xf.invert()
            * Point::new(
                T::from_f64(world_x).unwrap(),
                T::from_f64(world_y).unwrap(),
                T::from_f64(-1.0f64).unwrap(),
            );
        let origin = &self.xf.invert() * Point::new(T::zero(), T::zero(), T::zero());

        Ray::new(origin, (pixel - origin).normalize())
    }

    pub(crate) fn render<const W: usize, const H: usize>(
        &self,
        world: &World<T>,
        canvas: &mut Canvas<W, H>,
    ) {
        for y in 0..self.vsize {
            for x in 0..self.hsize {
                //  debug!(x, y);
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
        shape::{Shape, Sphere},
        vec::*,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_camera() {
        let c: Camera<f64> = Camera::new(200, 125, std::f64::consts::PI / 2.0);
        assert_relative_eq!(c.pixel_size, 0.01, epsilon = 0.001);

        let c: Camera<f64> = Camera::new(125, 200, std::f64::consts::PI / 2.0);
        assert_relative_eq!(c.pixel_size, 0.01, epsilon = 0.001);

        let c: Camera<f64> = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let r = c.ray_for_pixel(100, 50);
        assert_relative_eq!(r.o, Point::<f64>::new(0, 0, 0), epsilon = 0.0001);
        assert_relative_eq!(r.d, Vector::<f64>::new(0, 0, -1), epsilon = 0.0001);

        let r = c.ray_for_pixel(0, 0);
        assert_relative_eq!(r.o, Point::<f64>::new(0, 0, 0));
        assert_relative_eq!(
            r.d,
            Vector::<f64>::new(0.66519, 0.33259, -0.66851),
            epsilon = 0.00001
        );

        let mut c: Camera<f64> = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let s2 = 2.0f64.sqrt() / 2.0;
        c.xf = TransformBuilder::new()
            .translate(0, -2, 5)
            .rotation_y(std::f64::consts::PI / 4.0)
            .finish();
        let r = c.ray_for_pixel(100, 50);
        assert_relative_eq!(r.o, Point::<f64>::new(0, 2, -5));
        assert_relative_eq!(r.d, Vector::<f64>::new(s2, 0., -s2), epsilon = 0.00001);

        let xf = Camera::make_view_transform(
            Point::<f64>::new(0, 0, 0),
            Point::<f64>::new(0, 0, -1),
            Vector::<f64>::new(0, 1, 0),
        );
        assert_relative_eq!(xf, Matrix::identity());

        let xf = Camera::make_view_transform(
            Point::<f64>::new(0, 0, 0),
            Point::<f64>::new(0, 0, 1),
            Vector::<f64>::new(0, 1, 0),
        );
        assert_relative_eq!(xf, Matrix::make_scaling(-1, 1, -1));

        let xf = Camera::make_view_transform(
            Point::<f64>::new(0, 0, 8),
            Point::<f64>::new(0, 0, 0),
            Vector::<f64>::new(0, 1, 0),
        );
        assert_relative_eq!(xf, Matrix::make_translation(0, 0, -8));

        let xf = Camera::make_view_transform(
            Point::<f64>::new(1, 3, 2),
            Point::<f64>::new(4, -2, 8),
            Vector::<f64>::new(1, 1, 0),
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
        let c = Camera::new_with_transform(
            400,
            400,
            1.5,
            Point::<f64>::new(0, 0, -3),
            Point::<f64>::new(0, 0, 0),
            Vector::<f64>::new(0, 1, 0),
        );

        let w: World<f64> = World::default();
        println!("{}", w.describe());
    }
}

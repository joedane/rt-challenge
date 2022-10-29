use crate::vec::{Matrix, TransformBuilder};
use crate::MyFloat;
use crate::Point;
use crate::Ray;

struct Camera<T> {
    hsize: usize,
    vsize: usize,
    half_width: f64,
    half_height: f64,
    fov: f64,
    pixel_size: f64,
    xf: Matrix<T>,
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

    pub (crate) fn set_transform(&mut self, xf: Matrix<T>) {
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
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::vec::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_camera() {
        let c: Camera<f64> = Camera::new(200, 125, std::f64::consts::PI / 2.0);
        assert_relative_eq!(c.pixel_size, 0.01, epsilon = 0.001);

        let c: Camera<f64> = Camera::new(125, 200, std::f64::consts::PI / 2.0);
        assert_relative_eq!(c.pixel_size, 0.01, epsilon = 0.001);

        let c: Camera<f64> = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let r = c.ray_for_pixel(100, 50);
        assert_relative_eq!(r.o, Point::<f64>::new(0, 0, 0), epsilon=0.0001);
        assert_relative_eq!(r.d, Vector::<f64>::new(0, 0, -1), epsilon=0.0001);
   
        let r = c.ray_for_pixel(0, 0);
        assert_relative_eq!(r.o, Point::<f64>::new(0, 0, 0));
        assert_relative_eq!(r.d, Vector::<f64>::new(0.66519, 0.33259, -0.66851), epsilon=0.00001);
     
        let mut c: Camera<f64> = Camera::new(201, 101, std::f64::consts::PI / 2.0);
        let s2 = 2.0f64.sqrt() / 2.0;
        c.xf = TransformBuilder::new().translate(0, -2, 5).rotation_y(std::f64::consts::PI / 4.0).finish();
        let r = c.ray_for_pixel(100, 50);
        assert_relative_eq!(r.o, Point::<f64>::new(0, 2, -5));
        assert_relative_eq!(r.d, Vector::<f64>::new(s2, 0., -s2), epsilon=0.00001);

    }
}

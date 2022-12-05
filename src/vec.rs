use approx::{AbsDiffEq, RelativeEq};
use itertools::join;
use num_integer::Roots;
use std::fmt::Display;

use crate::{MyFloat2, TOLERANCE};

#[derive(Debug, Copy, Clone)]
pub struct Point {
    pub x: MyFloat2,
    pub y: MyFloat2,
    pub z: MyFloat2,
}

#[derive(Debug, Copy, Clone)]
pub struct Vector {
    pub x: MyFloat2,
    pub y: MyFloat2,
    pub z: MyFloat2,
}

impl Point {
    pub fn new<P: Into<MyFloat2>>(x: P, y: P, z: P) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    pub fn at_origin() -> Self {
        Self::new(0, 0, 0)
    }
}

impl AbsDiffEq for Point {
    type Epsilon = MyFloat2;

    fn default_epsilon() -> Self::Epsilon {
        TOLERANCE
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon)
            && self.y.abs_diff_eq(&other.y, epsilon)
            && self.z.abs_diff_eq(&other.z, epsilon)
    }
}
impl RelativeEq for Point {
    fn default_max_relative() -> Self::Epsilon {
        TOLERANCE
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_relative)
            && self.y.relative_eq(&other.y, epsilon, max_relative)
            && self.z.relative_eq(&other.z, epsilon, max_relative)
    }
}

impl AbsDiffEq for Vector {
    type Epsilon = MyFloat2;

    fn default_epsilon() -> Self::Epsilon {
        TOLERANCE
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon)
            && self.y.abs_diff_eq(&other.y, epsilon)
            && self.z.abs_diff_eq(&other.z, epsilon)
    }
}
impl RelativeEq for Vector {
    fn default_max_relative() -> Self::Epsilon {
        MyFloat2::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_relative)
            && self.y.relative_eq(&other.y, epsilon, max_relative)
            && self.z.relative_eq(&other.z, epsilon, max_relative)
    }
}

impl AbsDiffEq for Matrix {
    type Epsilon = MyFloat2;

    fn default_epsilon() -> Self::Epsilon {
        TOLERANCE
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.storage
            .iter()
            .zip(&other.storage)
            .all(|(this, that)| this.abs_diff_eq(that, epsilon))
    }
}

impl RelativeEq for Matrix {
    fn default_max_relative() -> Self::Epsilon {
        MyFloat2::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.storage
            .iter()
            .zip(&other.storage)
            .all(|(this, that)| this.relative_eq(that, epsilon, max_relative))
    }
}

impl std::cmp::PartialEq for Point {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
    }
}

impl std::ops::Neg for Point {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Point::new(-self.x, -self.y, -self.z)
    }
}

impl std::ops::Add<Vector> for Point {
    type Output = Point;

    fn add(self, rhs: Vector) -> Self::Output {
        Point::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub<Point> for Point {
    type Output = Vector;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Sub<Vector> for Point {
    type Output = Point;

    fn sub(self, rhs: Vector) -> Self::Output {
        Point::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Vector {
    pub fn new<P: Into<MyFloat2>>(x: P, y: P, z: P) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    pub fn reflect(&self, n: &Vector) -> Self {
        let dot_in_normal = Vector::dot(*self, *n);
        let dot_in_normal = dot_in_normal + dot_in_normal;

        *self - (*n * dot_in_normal)
    }
}

impl std::cmp::PartialEq for Vector {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
    }
}

impl std::ops::Neg for Vector {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vector::new(-self.x, -self.y, -self.z)
    }
}

impl std::ops::Add for Vector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Add<Point> for Vector {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<MyFloat2> for Vector {
    type Output = Self;

    fn mul(self, rhs: MyFloat2) -> Self::Output {
        Vector::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Div<MyFloat2> for Vector {
    type Output = Self;

    fn div(self, rhs: MyFloat2) -> Self::Output {
        Vector::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Vector {
    pub fn magnitude(&self) -> MyFloat2 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let m = self.magnitude();
        Vector::new(self.x / m, self.y / m, self.z / m)
    }

    pub fn dot(a: Self, b: Self) -> MyFloat2 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    pub fn cross(a: Self, b: Self) -> Self {
        Vector::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }
}

/**
 * A 4x4 generic Matrix
 */
#[derive(Debug)]
pub struct Matrix {
    size: usize,
    storage: Vec<MyFloat2>,
}

impl Matrix {
    pub fn new<P: Into<MyFloat2>>(v: Vec<P>) -> Self {
        let size = v.len().sqrt();
        Matrix {
            storage: v.into_iter().map(|v| v.into()).collect(),
            size,
        }
    }

    pub fn identity() -> Self {
        let size: usize = 4;
        let mut v = vec![0.0; size * size];
        for i in 0..size {
            *(v.get_mut((i % size) + i * size).unwrap()) = 1.0;
        }
        Matrix::new(v)
    }

    pub fn transpose(&self) -> Self {
        let mut v = vec![0.0; self.storage.len()];
        for row in 0..self.size {
            for col in 0..self.size {
                *(v.get_mut(row * self.size + col).unwrap()) = self.get(col, row);
            }
        }
        Matrix::new(v)
    }

    pub fn submatrix(&self, row: usize, col: usize) -> Matrix {
        assert!(self.size > 0);
        let mut v = Vec::with_capacity((self.size - 1) * (self.size - 1));
        for i in 0..self.size {
            if i != row {
                for j in 0..self.size {
                    if j != col {
                        v.push(self.get(i, j));
                    }
                }
            }
        }
        Matrix::new(v)
    }

    pub fn determinant(&self) -> MyFloat2 {
        if self.size == 2 {
            self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
        } else {
            let mut det: MyFloat2 = 0.0;
            for i in 0..self.size {
                det += self.get(0, i) * self.cofactor(0, i);
            }
            det
        }
    }

    /**
     * see https://stackoverflow.com/questions/71185365/method-bounded-by-a-const-generic-expression-does-not-satisfy-trait-bound
     */
    fn minor(&self, row: usize, col: usize) -> MyFloat2 {
        self.submatrix(row, col).determinant()
    }

    fn cofactor(&self, row: usize, col: usize) -> MyFloat2 {
        let m = self.minor(row, col);
        if (row + col) % 2 == 0 {
            m
        } else {
            -m
        }
    }

    pub fn invert(&self) -> Matrix {
        let det = self.determinant();
        if det == 0.0 {
            panic!();
        }
        let mut v: Vec<MyFloat2> = Vec::with_capacity(self.size);
        for row in 0..self.size {
            for col in 0..self.size {
                let c = self.cofactor(row, col);
                v.push(c / det);
            }
        }
        Matrix::new(v).transpose()
    }

    fn get(&self, row: usize, col: usize) -> MyFloat2 {
        self.storage[row * self.size + col]
    }

    fn set(&mut self, row: usize, col: usize, val: MyFloat2) {
        self.storage[row * self.size + col] = val;
    }

    pub fn make_translation<P: Into<MyFloat2>>(x: P, y: P, z: P) -> Matrix {
        let mut m = Matrix::identity();
        m.set(0, 3, x.into());
        m.set(1, 3, y.into());
        m.set(2, 3, z.into());
        m
    }

    pub fn make_scaling<P: Into<MyFloat2>>(x: P, y: P, z: P) -> Matrix {
        let mut m = Matrix::identity();
        m.set(0, 0, x.into());
        m.set(1, 1, y.into());
        m.set(2, 2, z.into());
        m
    }

    pub fn make_rotation_x<P: Into<MyFloat2>>(r: P) -> Matrix {
        let mut m = Matrix::identity();
        let r: MyFloat2 = r.into();
        m.set(1, 1, r.cos());
        m.set(2, 2, r.cos());
        m.set(1, 2, -r.sin());
        m.set(2, 1, r.sin());
        m
    }

    pub fn make_rotation_y<P: Into<MyFloat2>>(r: P) -> Matrix {
        let mut m = Matrix::identity();
        let r: MyFloat2 = r.into();
        m.set(0, 0, r.cos());
        m.set(0, 2, r.sin());
        m.set(2, 0, -r.sin());
        m.set(2, 2, r.cos());
        m
    }

    pub fn make_rotation_z<P: Into<MyFloat2>>(r: P) -> Matrix {
        let mut m = Matrix::identity();
        let r: MyFloat2 = r.into();
        m.set(0, 0, r.cos());
        m.set(0, 1, -r.sin());
        m.set(1, 0, r.sin());
        m.set(1, 1, r.cos());
        m
    }

    pub fn make_shearing<P: Into<MyFloat2>>(xy: P, xz: P, yx: P, yz: P, zx: P, zy: P) -> Matrix {
        let mut m = Matrix::identity();
        m.set(0, 1, xy.into());
        m.set(0, 2, xz.into());
        m.set(1, 0, yx.into());
        m.set(1, 2, yz.into());
        m.set(2, 0, zx.into());
        m.set(2, 1, zy.into());
        m
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.size {
            let joined = join(
                self.storage[row * self.size..(row + 1) * self.size]
                    .iter()
                    .map(|&val| format!("{:.4}", val)),
                "|",
            );
            writeln!(f, "{}", joined)?;
        }
        write!(f, "")
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
    }
}

impl std::ops::Mul<Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        let nx = self.get(0, 0) * rhs.x + self.get(0, 1) * rhs.y + self.get(0, 2) * rhs.z;
        let ny = self.get(1, 0) * rhs.x + self.get(1, 1) * rhs.y + self.get(1, 2) * rhs.z;
        let nz = self.get(2, 0) * rhs.x + self.get(2, 1) * rhs.y + self.get(2, 2) * rhs.z;
        Vector::new(nx, ny, nz)
    }
}

impl std::ops::Mul<Point> for &Matrix {
    type Output = Point;

    fn mul(self, rhs: Point) -> Self::Output {
        let nx = self.get(0, 0) * rhs.x
            + self.get(0, 1) * rhs.y
            + self.get(0, 2) * rhs.z
            + self.get(0, 3);
        let ny = self.get(1, 0) * rhs.x
            + self.get(1, 1) * rhs.y
            + self.get(1, 2) * rhs.z
            + self.get(1, 3);
        let nz = self.get(2, 0) * rhs.x
            + self.get(2, 1) * rhs.y
            + self.get(2, 2) * rhs.z
            + self.get(2, 3);
        Point::new(nx, ny, nz)
    }
}

impl std::ops::Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut v = vec![];
        for row in 0..self.size {
            for col in 0..self.size {
                let mut val = 0.0;
                for i in 0..self.size {
                    val += self.get(row, i) * rhs.get(i, col);
                }
                v.push(val);
            }
        }
        Matrix::new(v)
    }
}

pub struct TransformBuilder {
    xf: Matrix,
}

impl TransformBuilder {
    pub fn new() -> Self {
        TransformBuilder {
            xf: Matrix::identity(),
        }
    }

    pub fn translate<P: Into<MyFloat2>>(self, x: P, y: P, z: P) -> Self {
        TransformBuilder {
            xf: &Matrix::make_translation(x, y, z) * &self.xf,
        }
    }

    pub fn scaling<P: Into<MyFloat2>>(self, x: P, y: P, z: P) -> Self {
        TransformBuilder {
            xf: &Matrix::make_scaling(x, y, z) * &self.xf,
        }
    }

    pub fn shearing<P: Into<MyFloat2>>(self, xy: P, xz: P, yx: P, yz: P, zx: P, zy: P) -> Self {
        TransformBuilder {
            xf: &Matrix::make_shearing(xy, xz, yx, yz, zx, zy) * &self.xf,
        }
    }

    pub fn rotation_x<P: Into<MyFloat2>>(self, r: P) -> Self {
        TransformBuilder {
            xf: &Matrix::make_rotation_x(r) * &self.xf,
        }
    }

    pub fn rotation_y<P: Into<MyFloat2>>(self, r: P) -> Self {
        TransformBuilder {
            xf: &Matrix::make_rotation_y(r) * &self.xf,
        }
    }

    pub fn rotation_z<P: Into<MyFloat2>>(self, r: P) -> Self {
        TransformBuilder {
            xf: &Matrix::make_rotation_z(r) * &self.xf,
        }
    }

    pub fn finish(self) -> Matrix {
        self.xf
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use num_traits::Inv;

    #[test]
    fn test_eq1() {
        assert_eq!(Point::new(2., 2., 2.), Point::new(2., 2., 2.));
    }

    #[test]
    fn test_add1() {
        assert_eq!(
            Point::new(2., 2., 2.),
            Vector::new(1., 1., 1.) + Point::new(1., 1., 1.)
        );
        assert_eq!(
            Point::new(2., 2., 2.),
            Point::new(1., 1., 1.) + Vector::new(1., 1., 1.)
        );
    }

    #[test]
    fn test_sub1() {
        assert_eq!(
            Vector::new(-2., -4., -6.),
            Point::new(3., 2., 1.) - Point::new(5., 6., 7.)
        );
        assert_eq!(
            Point::new(-2., -4., -6.),
            Point::new(3., 2., 1.) - Vector::new(5., 6., 7.)
        );
        assert_eq!(
            Vector::new(-2., -4., -6.),
            Vector::new(3., 2., 1.) - Vector::new(5., 6., 7.)
        );
    }

    #[test]
    fn test_neg() {
        assert_eq!(Point::new(-2., -3., -4.), -Point::new(2., 3., 4.));
        assert_eq!(Vector::new(-2., -3., -4.), -Vector::new(2., 3., 4.));
    }

    #[test]
    fn test_mul1() {
        assert_eq!(Vector::new(3.5, -7., 10.5), Vector::new(1., -2., 3.) * 3.5);
        // assert_eq!(Vector::new(3.5, -7., 10.5), 3.5 * Vector::new(1., -2., 3.));
    }

    #[test]
    fn test_div1() {
        assert_eq!(Vector::new(0.5, -1., 1.5), Vector::new(1., -2., 3.) / 2.);
    }

    #[test]
    fn test_magnitude() {
        assert_eq!(1.0, Vector::new(1, 0, 0).magnitude());
        assert_eq!(1.0, Vector::new(0., 1., 0.).magnitude());
        assert_eq!(1.0, Vector::new(0., 0., 1.).magnitude());
        assert_eq!(
            14.0f64.sqrt(),
            Vector::new(1., 2., 3.).magnitude()
        );
    }

    #[test]
    fn test_normalize() {
        assert_eq!(
            Vector::new(1., 0., 0.),
            Vector::new(4., 0., 0.).normalize()
        );
        let l = 14.0f64.sqrt().inv();
        assert_eq!(
            Vector::new(l, 2. * l, 3. * l),
            Vector::new(1., 2., 3.).normalize()
        );
    }

    #[test]
    fn test_dot() {
        assert_eq!(
            20.,
            Vector::dot(Vector::new(1., 2., 3.), Vector::new(2., 3., 4.))
        );
    }

    #[test]
    fn test_cross() {
        let (v1, v2) = (
            Vector::new(1., 2., 3.),
            Vector::new(2., 3., 4.),
        );

        assert_eq!(Vector::cross(v1, v2), Vector::new(-1., 2., -1.));
        assert_eq!(Vector::cross(v2, v1), Vector::new(1., -2., 1.));
    }

    #[test]
    fn test_matrix_eq() {
        let m1: Matrix = Matrix::new(vec![
            1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
        ]);
        let m2: Matrix = Matrix::new(vec![
            1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
        ]);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_matrix_mul() {
        let m1: Matrix = Matrix::new(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 8., 7., 6., 5., 4., 3., 2.,
        ]);
        let m2: Matrix = Matrix::new(vec![
            -2., 1., 2., 3., 3., 2., 1., -1., 4., 3., 6., 5., 1., 2., 7., 8.,
        ]);
        let m3: Matrix = Matrix::new(vec![
            20., 22., 50., 48., 44., 54., 114., 108., 40., 58., 110., 102., 16., 26., 46., 42.,
        ]);
        assert_eq!(m3, &m1 * &m2)
    }

    #[test]
    fn test_tuple_mul() {
        let m: Matrix = Matrix::new(vec![
            1., 2., 3., 4., 2., 4., 4., 2., 8., 6., 4., 1., 0., 0., 0., 1.,
        ]);
        let p: Point = Point::new(1., 2., 3.);
        let p_check: Point = Point::new(18., 24., 33.);
        assert_eq!(p_check, &m * p);

        let v: Vector = Vector::new(1., 2., 3.);
        assert_eq!(v, &Matrix::identity() * v);
    }

    #[test]
    fn test_identity() {
        let m: Matrix = Matrix::identity();
        assert_eq!(
            m,
            Matrix::new(vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.
            ])
        );
    }

    #[test]
    fn test_transpose() {
        let m1: Matrix = Matrix::new(vec![
            0., 9., 3., 0., 9., 8., 0., 8., 1., 8., 5., 3., 0., 0., 5., 8.,
        ]);
        let m2: Matrix = Matrix::new(vec![
            0., 9., 1., 0., 9., 8., 8., 0., 3., 0., 5., 5., 0., 8., 3., 8.,
        ]);
        assert_eq!(m2, m1.transpose());
    }

    #[test]
    fn test_submatrix() {
        let m1: Matrix = Matrix::new(vec![1., 5., 0., -3., 2., 7., 0., 6., -3.]);
        let m2: Matrix = Matrix::new(vec![-3., 2., 0., 6.]);
        assert_eq!(m1.submatrix(0, 2), m2);
    }

    #[test]
    fn test_determinant() {
        let m: Matrix = Matrix::new(vec![1., 5., -3., 2.]);
        assert_eq!(m.determinant(), 17.);
    }

    #[test]
    fn test_cofactor() {
        let m: Matrix = Matrix::new(vec![3., 5., 0., 2., -1., -7., 6., -1., 5.]);
        assert_eq!(-12., m.cofactor(0, 0));
        assert_eq!(-25., m.cofactor(1, 0));
    }

    fn check_matrix(
        m: Matrix,
        m_check: Matrix,
    ) -> Result<(), String> {
        for row in 0..4 {
            for col in 0..4 {
                if (m.get(row, col) - m_check.get(row, col)).abs() > TOLERANCE {
                    panic!(
                        "comparison failed at ({}, {}), expected {}, got {}",
                        row,
                        col,
                        m_check.get(row, col),
                        m.get(row, col)
                    )
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_invert() {
        #[rustfmt::skip]
        let m: Matrix = Matrix::new(vec![
            -5, 2, 6, -8,
            1, -5, 1, 8,
            7, 7, -6, -7,
            1, -3, 7, 4,
        ]);
        #[rustfmt::skip]
        let mi_check: Matrix = Matrix::new(vec![
            0.21805, 0.45113, 0.24060, -0.04511, 
            -0.80827, -1.45677, -0.44361, 0.52068, 
            -0.07895, -0.22368, -0.05263, 0.19737, 
            -0.52256, -0.81391, -0.30075, 0.30639,
        ]);
        let mi = m.invert();
        assert_eq!(532.0, m.determinant());
        assert_eq!(-160., m.cofactor(2, 3));
        assert_eq!(-160. / 532., mi.get(3, 2));
        assert_eq!(105., m.cofactor(3, 2));
        assert_eq!(105. / 532., mi.get(2, 3));
        check_matrix(mi, mi_check).unwrap();

        #[rustfmt::skip]
        let m: Matrix = Matrix::new(vec![
            8, -5, 9, 2,
            7, 5, 6, 1,
            -6, 0, 9, 6,
            -3, 0, -9, -4,
        ]);

        #[rustfmt::skip]
        let mi_check: Matrix = Matrix::new(vec![
            -0.15385, -0.15385, -0.28205, -0.53846,
            -0.07692, 0.12308, 0.02564, 0.03077,
            0.35897, 0.35897, 0.43590, 0.92308, 
            -0.69231, -0.69231, -0.76923, -1.92308,
        ]);
        check_matrix(m.invert(), mi_check).unwrap();

        #[rustfmt::skip]
        let m: Matrix = Matrix::new(vec![
            9, 3, 0, 9,
            -5, -2, -6, -3,
            -4, 9, 6, 4,
            -7, 6, 6, 2,
        ]);
        #[rustfmt::skip]
        let mi_check: Matrix = Matrix::new(vec![
            -0.04074, -0.07778, 0.14444, -0.22222, 
            -0.07778, 0.03333, 0.36667, -0.33333, 
            -0.02901, -0.14630, -0.10926, 0.12963, 
            0.17778, 0.06667, -0.26667, 0.33333,
        ]);
        check_matrix(m.invert(), mi_check).unwrap();

        #[rustfmt::skip]
        let a: Matrix = Matrix::new(vec![
            3, -9, 7, 3,
            3, -8, 2, -9,
            -4, 4, 4, 1,
            -6, 5, -1, 1,
        ]);

        #[rustfmt::skip]
        let b: Matrix = Matrix::new(vec![
            8, 2, 2, 2, 
            3, -1, 7, 0,
            7, 0, 5, 4,
            6, -2, 0, 5,
        ]);

        let c = &a * &b;
        let bi = b.invert();
        let check_a = &c * &bi;
        check_matrix(check_a, a).unwrap();
    }

    #[test]
    fn test_translation() {
        let x: Matrix = Matrix::make_translation(5, -3, 2);
        let p: Point = Point::new(-3, 4, 5);
        assert_eq!(&x * p, Point::new(2, 1, 7));

        let xi = x.invert();
        assert_eq!(&xi * p, Point::new(-8, 7, 3));

        let v: Vector = Vector::new(-3, 4, 5);
        assert_eq!(&x * v, v);
    }

    #[test]
    fn test_scaling() {
        let x: Matrix = Matrix::make_scaling(2, 3, 4);
        let p: Point = Point::new(-4, 6, 8);
        assert_eq!(&x * p, Point::new(-8, 18, 32));

        let v: Vector = Vector::new(-4, 6, 8);
        assert_eq!(&x * v, Vector::new(-8, 18, 32));
    }

    #[test]
    fn test_rotation() {
        let half: Matrix = Matrix::make_rotation_x(std::f64::consts::PI / 4.0);
        let full: Matrix = Matrix::make_rotation_x(std::f64::consts::PI / 2.0);
        let s2 = 2.0f64.sqrt() / 2.0;
        let p: Point = Point::new(0, 1, 0);
        assert_relative_eq!(&half * p, Point::new(0., s2, s2));
        assert_relative_eq!(&full * p, Point::new(0, 0, 1));

        let half: Matrix = Matrix::make_rotation_y(std::f64::consts::PI / 4.0);
        let full: Matrix = Matrix::make_rotation_y(std::f64::consts::PI / 2.0);
        let p: Point = Point::new(0, 0, 1);
        assert_relative_eq!(&half * p, Point::new(s2, 0., s2));
        assert_relative_eq!(&full * p, Point::new(1, 0, 0));

        let half: Matrix = Matrix::make_rotation_z(std::f64::consts::PI / 4.0);
        let full: Matrix = Matrix::make_rotation_z(std::f64::consts::PI / 2.0);
        let p: Point = Point::new(0, 1, 0);
        assert_relative_eq!(&half * p, Point::new(-s2, s2, 0.));
        assert_relative_eq!(&full * p, Point::new(-1, 0, 0));
    }

    #[test]
    fn test_shearing() {
        let x: Matrix = Matrix::make_shearing(1, 0, 0, 0, 0, 0);
        let p: Point = Point::new(2, 3, 4);
        assert_eq!(&x * p, Point::new(5, 3, 4));

        let x: Matrix = Matrix::make_shearing(0, 1, 0, 0, 0, 0);
        let p: Point = Point::new(2, 3, 4);
        assert_eq!(&x * p, Point::new(6, 3, 4));

        let x: Matrix = Matrix::make_shearing(0, 0, 1, 0, 0, 0);
        let p: Point = Point::new(2, 3, 4);
        assert_eq!(&x * p, Point::new(2, 5, 4));

        let x: Matrix = Matrix::make_shearing(0, 0, 0, 1, 0, 0);
        let p: Point = Point::new(2, 3, 4);
        assert_eq!(&x * p, Point::new(2, 7, 4));

        let x: Matrix = Matrix::make_shearing(0, 0, 0, 0, 1, 0);
        let p: Point = Point::new(2, 3, 4);
        assert_eq!(&x * p, Point::new(2, 3, 6));

        let x: Matrix = Matrix::make_shearing(0, 0, 0, 0, 0, 1);
        let p: Point = Point::new(2, 3, 4);
        assert_eq!(&x * p, Point::new(2, 3, 7));
    }

    #[test]
    fn test_builder() {
        let xf: Matrix = TransformBuilder::new()
            .rotation_x(std::f64::consts::PI / 2.)
            .scaling(5, 5, 5)
            .translate(10, 5, 7)
            .finish();
        let p: Point = Point::new(1, 0, 1);
        assert_eq!(&xf * p, Point::new(15, 0, 7));
    }

    #[test]
    fn test_reflect() {
        let v: Vector = Vector::new(1, -1, 0);
        let n: Vector = Vector::new(0, 1, 0);
        assert_eq!(v.reflect(&n), Vector::new(1, 1, 0));

        let s2 = 2.0f64.sqrt() / 2.0;
        let v: Vector = Vector::new(0, -1, 0);
        let n: Vector = Vector::new(s2, s2, 0.);
        assert_relative_eq!(v.reflect(&n), Vector::new(1, 0, 0));
    }
}

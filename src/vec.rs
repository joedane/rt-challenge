use itertools::join;
use num_traits::float::Float;
use std::{fmt::Display, ops::AddAssign};

#[derive(Debug)]
pub struct Point<T: Float> {
    x: T,
    y: T,
    z: T,
}

#[derive(Debug)]
pub struct Vector<T: Float> {
    x: T,
    y: T,
    z: T,
}

impl<T: Float> Point<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Float> std::cmp::PartialEq for Point<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
    }
}

impl<T: Float> std::ops::Neg for Point<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Point::new(-self.x, -self.y, -self.z)
    }
}

impl<T: Float> std::ops::Add<Vector<T>> for Point<T> {
    type Output = Point<T>;

    fn add(self, rhs: Vector<T>) -> Self::Output {
        Point::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T: Float> std::ops::Sub<Point<T>> for Point<T> {
    type Output = Vector<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<T: Float> std::ops::Sub<Vector<T>> for Point<T> {
    type Output = Point<T>;

    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Point::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

/*
const EPSILON: f32 = 0.00001;

impl<T: Num + Copy + Signed> PartialEq for Point<T> {

    fn eq(&self, other: &Self) -> bool {
        let _diff = num_traits::sign::abs(self.x - other.x);
        false
    }
}
*/

impl<T: Float> Vector<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Float> std::cmp::PartialEq for Vector<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
    }
}

impl<T: Float> std::ops::Neg for Vector<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vector::new(-self.x, -self.y, -self.z)
    }
}

impl<T: Float> std::ops::Add for Vector<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: Float> std::ops::Add<Point<T>> for Vector<T> {
    type Output = Point<T>;

    fn add(self, rhs: Point<T>) -> Self::Output {
        Point::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<T: Float> std::ops::Sub for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, rhs: Vector<T>) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<T> std::ops::Mul<T> for Vector<T>
where
    T: Float,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Vector::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl<T> std::ops::Div<T> for Vector<T>
where
    T: Float,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Vector::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<T> Vector<T>
where
    T: Float,
{
    pub fn magnitude(&self) -> T {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let m = self.magnitude();
        Vector::new(self.x / m, self.y / m, self.z / m)
    }

    pub fn dot(a: Self, b: Self) -> T {
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

#[derive(Debug)]
pub struct Matrix<T: Float, const M: usize> {
    storage: [[T; M]; M],
}

impl<T: Float + std::ops::AddAssign, const M: usize> Matrix<T, M> {
    fn new(v: Vec<T>) -> Self {
        if v.len() != M * M {
            panic!();
        }
        let mut m = Matrix {
            storage: [[T::zero(); M]; M],
        };
        for i in 0..M {
            for j in 0..M {
                m.storage[j][i] = v[j * M + i]
            }
        }
        m
    }

    fn identity() -> Self {
        let mut v = vec![T::zero(); M * M];
        for i in 0..M {
            *(v.get_mut((i % M) + i * M).unwrap()) = T::one();
        }
        Matrix::new(v)
    }

    fn transpose(&self) -> Self {
        let mut v = vec![T::zero(); M * M];
        for row in 0..M {
            for col in 0..M {
                *(v.get_mut(row * M + col).unwrap()) = self.storage[col][row]
            }
        }
        Matrix::new(v)
    }

    fn submatrix(&self, row: usize, col: usize) -> Matrix<T, { M - 1 }> where [(); {M-1}]: {
        let mut v = vec![];
        for i in 0..M {
            if i != row {
                for j in 0..M {
                    if j != col {
                        v.push(self.get(i, j));
                    }
                }
            }
        }
        Matrix::new(v)
    }

    fn determinant(&self) -> T where [(); M-1]: {
        if M == 2 {
            let s = self.storage;
            return s[0][0]*s[1][1] - s[0][1]*s[1][0];
        } else {
            let det: T = T::zero();
            for i in 0..M {
                det += self.get(0, i) * self.cofactor(0, i);
            }
            det
        }
    }

    /**
     * see https://stackoverflow.com/questions/71185365/method-bounded-by-a-const-generic-expression-does-not-satisfy-trait-bound
     */
    fn minor(&self, row: usize, col: usize) -> T where [(); M-1]: {
        self.submatrix(row, col).determinant()
    }

    fn cofactor(&self, row: usize, col: usize) -> T where [(); M-1]: {
        let m = self.minor(row, col);
        if row + col % 2 == 0 {
            m
        } else {
            -m
        }
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.storage[row][col]
    }
}

impl<T: Float + Display, const M: usize> Display for Matrix<T, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..M {
            let joined = join(
                self.storage[row].iter().map(|&val| format!("{:.4}", val)),
                "|",
            );
            write!(f, "{}\n", joined)?;
        }
        write!(f, "")
    }
}

impl<T: Float, const M: usize> PartialEq for Matrix<T, M> {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
    }
}

impl<T: Float + AddAssign, const M: usize> std::ops::Mul for &Matrix<T, M> {
    type Output = Matrix<T, M>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut v = vec![];
        for row in 0..M {
            for col in 0..M {
                let mut val = T::zero();
                for i in 0..M {
                    val += self.get(row, i) * rhs.get(i, col);
                }
                v.push(val);
            }
        }
        Matrix::new(v)
    }
}
#[cfg(test)]
mod test {
    use num_traits::Inv;

    use super::*;

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
        assert_eq!(1.0, Vector::new(1., 0., 0.).magnitude());
        assert_eq!(1.0, Vector::new(0., 1., 0.).magnitude());
        assert_eq!(1.0, Vector::new(0., 0., 1.).magnitude());
        assert_eq!((14.0 as f64).sqrt(), Vector::new(1., 2., 3.).magnitude());
    }

    #[test]
    fn test_normalize() {
        assert_eq!(Vector::new(1., 0., 0.), Vector::new(4., 0., 0.).normalize());
        let l = (14.0 as f64).sqrt().inv();
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
        let (v1, v2) = (Vector::new(1., 2., 3.), Vector::new(2., 3., 4.));
    }

    #[test]
    fn test_matrix_eq() {
        let m1: Matrix<f64, 4> = Matrix::new(vec![
            1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
        ]);
        let m2: Matrix<f64, 4> = Matrix::new(vec![
            1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
        ]);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_matrix_mul() {
        let m1: Matrix<f64, 4> = Matrix::new(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 8., 7., 6., 5., 4., 3., 2.,
        ]);
        let m2: Matrix<f64, 4> = Matrix::new(vec![
            -2., 1., 2., 3., 3., 2., 1., -1., 4., 3., 6., 5., 1., 2., 7., 8.,
        ]);
        let m3: Matrix<f64, 4> = Matrix::new(vec![
            20., 22., 50., 48., 44., 54., 114., 108., 40., 58., 110., 102., 16., 26., 46., 42.,
        ]);
        assert_eq!(m3, &m1 * &m2)
    }

    #[test]
    fn test_identity() {
        let m: Matrix<f64, 4> = Matrix::identity();
        assert_eq!(
            m,
            Matrix::new(vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.
            ])
        );
    }

    #[test]
    fn test_transpose() {
        let m1: Matrix<f64, 4> = Matrix::new(vec![
            0., 9., 3., 0., 9., 8., 0., 8., 1., 8., 5., 3., 0., 0., 5., 8.,
        ]);
        let m2: Matrix<f64, 4> = Matrix::new(vec![
            0., 9., 1., 0., 9., 8., 8., 0., 3., 0., 5., 5., 0., 8., 3., 8.,
        ]);
        assert_eq!(m2, m1.transpose());
    }

    #[test]
    fn test_submatrix() {
        let m1: Matrix<f64, 3> = Matrix::new(vec![1., 5., 0., -3., 2., 7., 0., 6., -3.]);
        let m2: Matrix<f64, 2> = Matrix::new(vec![-3., 2., 0., 6.]);
        assert_eq!(m1.submatrix(0, 2), m2);
    }

    #[test]
    fn test_determinant() {
        let m: Matrix<f64, 2> = Matrix::new(vec![1., 5., -3., 2.]);
        assert_eq!(m.determinant(), 17.);
    }

    #[test]
    fn test_cofactor() {
        let m: Matrix<f64, 3> = Matrix::new(vec![3., 5., 0., 2., -1., -7., 6., -1., 5.]);
        assert_eq!(-12., m.cofactor(0, 0));
        assert_eq!(-25., m.cofactor(1, 0));
    }
}

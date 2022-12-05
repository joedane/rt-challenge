use num_traits::{Float, ToPrimitive};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Color {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
}

impl Color {
    pub const WHITE: Color = Color {
        red: 1.0,
        green: 1.0,
        blue: 1.0,
    };

    pub fn new<P: Into<f32>>(red: P, green: P, blue: P) -> Self {
        Self {
            red: red.into(),
            green: green.into(),
            blue: blue.into(),
        }
    }
}

impl std::ops::Add for Color {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Color::new(
            self.red + rhs.red,
            self.green + rhs.green,
            self.blue + rhs.blue,
        )
    }
}

impl std::ops::Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Color::new(
            self.red * rhs.red,
            self.green * rhs.green,
            self.blue * rhs.blue,
        )
    }
}

impl<T: Float> std::ops::Mul<T> for Color {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = <T as ToPrimitive>::to_f32(&rhs).unwrap();
        Color::new(rhs * self.red, self.green * rhs, self.blue * rhs)
    }
}

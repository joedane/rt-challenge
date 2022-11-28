use crate::color::Color;
use image::{ImageBuffer, Rgb};
use tracing::debug;

pub struct Canvas<const W: usize, const H: usize> {
    storage: [[Color; W]; H],
}

impl<const W: usize, const H: usize> std::fmt::Debug for Canvas<W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Canvas [{}, {}]", W, H))
    }
}
impl<const W: usize, const H: usize> Canvas<W, H> {
    pub fn new() -> Self {
        Canvas {
            storage: [[Color::new(0., 0., 0.); W]; H],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, c: Color) {
        self.storage[y][x] = c;
    }

    pub fn get(&self, x: usize, y: usize) -> Color {
        self.storage[y][x]
    }

    #[tracing::instrument]
    pub fn write<P: AsRef<std::path::Path> + std::fmt::Debug>(&self, p: P) {
        let mut buf = ImageBuffer::new(W as u32, H as u32);
        debug!("writing file ...\n");
        for (x, y, p) in buf.enumerate_pixels_mut() {
            let c = self.get(x as usize, y as usize);
            *p = Rgb([c.red as u8, c.green as u8, c.blue as u8]);
        }
        buf.save(p).unwrap();
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_write() {
        let mut c: Canvas<100, 200> = Canvas::new();
        for i in 0..100 {
            for j in 0..200 {
                c.set(i, j, Color::new(j as f32, 0., 0.));
            }
        }
        c.write("testfile.jpg");
    }
}

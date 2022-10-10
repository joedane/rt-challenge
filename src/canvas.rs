
use crate::color::Color;
use image::{ImageBuffer, Rgb};

pub struct Canvas<const W: usize, const H: usize> {
    storage: [[Color; W] ; H]
}

impl<const W: usize, const H: usize> Canvas<W, H> {

    fn new() -> Self {
        Canvas { storage: [[Color::new(0,0,0); W]; H]}    
    }

    fn foo(&self) {
        println!("width: {}", W);
        println!("height: {}", H);
    }

    fn set(&mut self, x: usize, y: usize, c: Color) {
        self.storage[y][x] = c;
    }

    fn get(&self, x: usize, y: usize) -> Color {
        self.storage[y][x]
    }

    fn write<P: AsRef<std::path::Path>>(&self, p: P) {
        let mut buf = ImageBuffer::new(W as u32, H as u32);
        for (x, y, p) in buf.enumerate_pixels_mut() {
            let c = self.get(x as usize, y as usize);
            *p = Rgb([c.red, c.green, c.blue]);
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
                c.set(i, j, Color::new(j as u8, 0, 0));
            }
        }
        c.foo();
        c.write("testfile.jpg");
    }
}


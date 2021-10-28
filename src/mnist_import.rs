use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;

use byteorder::{BigEndian, ReadBytesExt};

const IMAGE_MAGIC: u32 = 0x00000803;
const LABEL_MAGIC: u32 = 0x00000801;

pub const IMAGE_SIZE: u32 = 28;

pub type Float = f32;
pub type Data = Vec<(Vec<Float>, Vec<Float>)>;

//--------
// This file was written by serxka
//--------

fn parse_images<R: Read>(mut data: R) -> IoResult<Data> {
    // Magic number
    assert_eq!(IMAGE_MAGIC, data.read_u32::<BigEndian>()?);
    // Number of images
    let len = data.read_u32::<BigEndian>()?;
    // Size of images
    assert_eq!(IMAGE_SIZE, data.read_u32::<BigEndian>()?);
    assert_eq!(IMAGE_SIZE, data.read_u32::<BigEndian>()?);

    let mut images = Vec::with_capacity(len as usize);
    for _ in 0..len {
        let mut image = vec![0; (IMAGE_SIZE * IMAGE_SIZE) as usize];
        data.read_exact(&mut image)?;
        let image = image.into_iter().map(|x| x as Float / 255.0).collect();
        images.push((image, vec![0.0; 10]));
    }

    Ok(images)
}

fn parse_labels<R: Read>(data: &mut Data, mut labels: R) -> IoResult<()> {
    // Magic number
    assert_eq!(LABEL_MAGIC, labels.read_u32::<BigEndian>()?);
    // Number of labels
    assert_eq!(data.len() as u32, labels.read_u32::<BigEndian>()?);

    let mut l = vec![0u8; data.len()];
    labels.read_exact(&mut l)?;

    for (d, l) in data.iter_mut().zip(l.iter()) {
        d.1[*l as usize] = 1.0;
    }

    Ok(())
}

pub fn parse<R: Read>(images: R, labels: R) -> IoResult<Data> {
    let mut data = parse_images(images)?;
    parse_labels(&mut data, labels)?;

    Ok(data)
}

pub fn parse_files<P: AsRef<Path>>(images: P, labels: P) -> IoResult<Data> {
    let image = File::open(images)?;
    let labels = File::open(labels)?;

    parse(image, labels)
}

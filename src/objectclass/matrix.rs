pub struct Matrix{
    content: Vec<Vec<f64>>,
}

impl Matrix{
    pub fn new(content: Vec<Vec<f64>>)->Matrix{
        Matrix { content: content }
    }

    pub fn getdim(&mut self)->i32{
        self.content.len() as i32
    }
}

impl Drop for Matrix{
    fn drop(&mut self) {
        
    }
}
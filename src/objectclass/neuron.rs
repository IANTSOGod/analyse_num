pub struct Neuron{
    weight:f64
}

impl Neuron{
    pub fn new(weight:f64)->Neuron{
        Neuron{
            weight: weight
        }
    }

    pub fn printdata(&mut self){
        println!("weight:{}",self.weight);
    }
}

impl Drop for Neuron{
    fn drop(&mut self) {
    }
}
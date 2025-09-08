mod objectclass;
mod operationclass;

use crate::objectclass::neuron::Neuron;

fn main() {
    let mut n1=Neuron::new(5.0);
    n1.printdata();
}

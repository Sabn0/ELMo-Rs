
// use device..

pub mod training {

    use std::error::Error;
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    use tch::{Tensor, Kind};
    use tch::data::Iter2;
    use tch::Device;
    use tch::nn::{VarStore, ModuleT, Optimizer, Adam, OptimizerConfig};
    use crate::ELMo;

    pub trait TrainModel {
        
        // train forces (x,y) labels for now (classification)
        fn train(&self, trainset_iter: &mut Iter2, devset_iter: &mut Option<Iter2>, learning_rate: f64, max_iter: i64, vocab_size: i64, model: &impl ModuleT) -> Result<(), Box<dyn Error>>;
        fn validate(&self, devset_iter: &mut Iter2, model: &impl ModuleT, vocab_size: i64) -> (f64, f64);
        fn step(&self, xs: Tensor, ys: Tensor, vocab_size: i64, model: &impl ModuleT, opt: Option<&mut Optimizer>, loss: &mut f64, accuracy: &mut f64);       fn predict(&self, targets: &Tensor, logits: &Tensor) -> f64;
        fn init_optimizer(&self, vars: &VarStore, learning_rate: f64) -> Optimizer;
        fn init_vars(&self) -> VarStore;
        

        fn log(&self, out_path: &str, items: Vec<String>) -> Result<(), Box<dyn Error>> {
            
            let out_string = items.join("\t");
            //out_string += items.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("\t");
            let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(out_path)?;
            
            writeln!(file, "{}", out_string)?;
            Ok(())

        }
        fn break_early(&self, _epoch_loss: f64, _dev_loss: f64) -> bool {
            false
        } 
    }

    struct elmo_trainer {
        trainset: Iter2,
        devset_iter: Option<Iter2>,
        learning_rate: f64,
        max_iter: i64,
        model: ELMo,
        device: Device
    }

    impl elmo_trainer {
        pub fn new() -> Self {
            todo!()
        }

        pub fn save() {

        }

        pub fn run() {

        }
    }

    impl TrainModel for elmo_trainer {
        
        fn train(&self, trainset_iter: &mut Iter2, devset_iter: &mut Option<Iter2>, learning_rate: f64, max_iter: i64, vocab_size: i64, model: &impl ModuleT) -> Result<(), Box<dyn Error>> {
            
            let vars = self.init_vars();
            let mut opt = self.init_optimizer(&vars, learning_rate);

            for _epoch in 0..max_iter {

                let mut epoch_loss = 0.0;
                let mut epoch_accuracy = 0.0;

                for (xs, ys) in trainset_iter.shuffle().into_iter().to_device(vars.device()) {

                    // xs of shape (batch_size, sequence_length, char_vocab_size)
                    // ys of shape (batch_size, sequence_length)
                    
                    self.step(xs, ys, vocab_size, model, Some(&mut opt), &mut epoch_loss, &mut epoch_accuracy);
                }

                if devset_iter.is_some() {

                    let dev_iter = devset_iter.as_mut().unwrap();
                    let (dev_loss, dev_accuracy) = self.validate(dev_iter, model, vocab_size);
                    
                    if self.break_early(epoch_loss, dev_loss) {
                        break;
                    }
                }


            }

            Ok(())

        
        }

        fn step(&self, xs: Tensor, ys: Tensor, vocab_size: i64, model: &impl ModuleT, opt: Option<&mut Optimizer>, loss: &mut f64, accuracy: &mut f64) {

            let train_mode = match &opt {
                Some(_) => true,
                None => false
            };

            let logits = model.forward_t(&xs, train_mode); // move throught model...
            // logits of shape (batch_Size, sequence_lngth, token_vocab_size)

            // logits and y should move to 2dim for cross entropy loss. This is another reason for batch_size =1...
            let logits = logits.reshape(&[-1, vocab_size]);
            let targets = ys.reshape(&[-1]).totype(Kind::Int64);
            let batch_loss = logits.cross_entropy_for_logits(&targets);

            if train_mode {
                opt.unwrap().backward_step(&batch_loss);
            }

            *loss += <f64>::from(batch_loss.mean(Kind::Float));
            *accuracy += self.predict(&targets, &logits);

        }

        fn validate(&self, devset_iter: &mut Iter2, model: &impl ModuleT, vocab_size: i64) -> (f64, f64) {

            let mut loss = 0.0;
            let mut accuracy = 0.0;

            for (xs, ys) in devset_iter.shuffle().into_iter().to_device(self.device) {

                // xs of shape (batch_size, sequence_length, char_vocab_size)
                // ys of shape (batch_size, sequence_length)                
                self.step(xs, ys, vocab_size, model, None, &mut loss, &mut accuracy);

            }

            (loss, accuracy)

        }

        fn predict(&self, targets: &Tensor, logits: &Tensor) -> f64 {

            // targets are of shape (batch_size * sequence_length)
            // logits are of shape (batch_size * sequence_length, vocab_size)
            let dims = targets.internal_shape_as_tensor();
            let dims = Vec::<i64>::from(dims);
            let n_samples = dims[0];

            // create predictions from logits based on argmax
            let predictions = logits.argmax(1, false);
            let accuracy = self.model.batch_accuracy_for_logits(&predictions, &targets, self.device, n_samples);
            accuracy
        }

        fn init_optimizer(&self, vars: &VarStore, learning_rate: f64) -> Optimizer {

            let opt: Optimizer = Adam::default().build(&vars, learning_rate).unwrap();
            opt

        }

        fn init_vars(&self) -> VarStore {

            let vars = VarStore::new(self.device);
            vars
        }
    }

}
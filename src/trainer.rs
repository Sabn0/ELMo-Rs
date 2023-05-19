
// use device..

pub mod training {

    use std::error::Error;
    use tch::{Tensor, Kind};
    use tch::data::Iter2;
    use tch::Device;
    use tch::nn::{VarStore, ModuleT, Optimizer, Adam, OptimizerConfig};
    use crate::ELMo;

    pub trait TrainModel {
        // train forces (x,y) labels for now (classification)
        fn train(&self, trainset_iter: &mut Iter2, devset_iter: &mut Option<Iter2>, learning_rate: f64, max_iter: i64, vocab_size: i64, model: &impl ModuleT) -> Result<(), Box<dyn Error>>;
        fn validate();
        fn predict(&self, targets: &Tensor, logits: &Tensor) -> f64;
        fn init_optimizer(&self, vars: &VarStore, learning_rate: f64) -> Optimizer;
        fn init_vars(&self) -> VarStore;
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
            
            let set_train = true;
            let vars = self.init_vars();
            let mut opt = self.init_optimizer(&vars, learning_rate);

            for _epoch in 0..max_iter {

                let mut loss = 0;
                let mut accuracy = 0.0;

                for (xs, ys) in trainset_iter.shuffle().into_iter().to_device(vars.device()) {

                    // xs of shape (batch_size, sequence_length, char_vocab_size)
                    // ys of shape (batch_size, sequence_length)
                    
                    let logits = model.forward_t(&xs, set_train); // move throught training...
                    // logits of shape (batch_Size, sequence_lngth, token_vocab_size)

                    // logits and y should move to 2dim for cross entropy loss. This is another reason for batch_size =1...
                    let logits = logits.reshape(&[-1, vocab_size]);
                    let targets = ys.reshape(&[-1]).totype(Kind::Int64);
                    let batch_loss = logits.cross_entropy_for_logits(&targets);
                    opt.backward_step(&batch_loss);

                    loss += batch_loss.mean(Kind::Float).numel();
                    accuracy += self.predict(&targets, &logits);

                }

                if devset_iter.is_some() {
                    todo!()
                }


            }

            Ok(())

        
        }


        fn validate() {
        todo!()
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
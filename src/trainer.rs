
// use device..

pub mod training {

    use std::error::Error;
    use std::fmt::Display;
    use std::ops::Add;
    use std::time::Instant;
    use tch::{Tensor, Kind};
    use tch::data::Iter2;
    use tch::nn::{VarStore, ModuleT, Optimizer, Adam, OptimizerConfig};
    use crate::{ELMo, Loader};

    pub trait TrainModel {
        
        // train forces (x,y) labels for now (classification)
        fn train(&self, trainset_iter: &mut Loader, devset_iter: &mut Option<Loader>, learning_rate: f64, max_iter: i64, model: &impl ModuleT, vars: &mut VarStore, clip_norm: f64) -> Result<(), Box<dyn Error>>;
        fn validate(&self, devset_iter: &mut Loader, model: &impl ModuleT, clip_norm: f64) -> (f64, f64);
        fn step(&self, xs: Tensor, ys: Tensor, model: &impl ModuleT, opt: Option<&mut Optimizer>, loss: &mut f64, accuracy: &mut f64, clip_norm: f64);       
        fn predict(&self, targets: &Tensor, logits: &Tensor) -> f64;
        fn init_optimizer(&self, vars: &VarStore, learning_rate: f64) -> Optimizer;
        
        fn break_early(&self, _epoch_loss: f64, _dev_loss: f64) -> bool { 
            false 
        }
        
        fn save_model(&self, out_path: &str, vars: &VarStore) -> Result<(), Box<dyn Error>> {
            Ok(vars.save(out_path)?)
        }
    }

    pub struct ElmoTrainer;

    impl ElmoTrainer {

        pub fn new() -> Self { Self {} }

        pub fn run_training(&self, trainset_iter: &mut Loader, devset_iter: &mut Option<Loader>, learning_rate: f64, max_iter: i64, model: ELMo, vars: &mut VarStore, clip_norm: f64) -> Result<(), Box<dyn Error>> {

            self.train(trainset_iter, devset_iter, learning_rate, max_iter, &model, vars, clip_norm)?;
            Ok(())
        }

        pub fn run_testing(&self, _testset_iter: &mut Iter2) {
            todo!()
        }

    }

    impl TrainModel for ElmoTrainer {
        
        fn train(&self, trainset_iter: &mut Loader, devset_iter: &mut Option<Loader>, learning_rate: f64, max_iter: i64, model: &impl ModuleT, vars: &mut VarStore, clip_norm: f64) -> Result<(), Box<dyn Error>> {
            
            let mut opt = self.init_optimizer(&vars, learning_rate);
            let mut train_progress = match devset_iter {
                Some(_) => TrainingProgress::init_with_dev(),
                None => TrainingProgress::init_no_dev()
            };
            
            for epoch in 0..max_iter {

                let timer = Instant::now();
                let mut total = 0.0;
                let mut epoch_loss = 0.0;
                let mut epoch_accuracy = 0.0;

                for (xs, ys) in trainset_iter.shuffle().to_stream().into_iter() {

                    // xs of shape (sequence_length, max_token_length)
                    // ys of shape (sequence_length)
                    self.step(xs, ys, model, Some(&mut opt), &mut epoch_loss, &mut epoch_accuracy, clip_norm);
                    total += 1.0;
                }

                // update training progress
                epoch_loss /= total;
                epoch_accuracy /= total;

                let mut progress_entry = TrainingProgress {
                    epoch: vec![epoch], epoch_loss: vec![epoch_loss], epoch_accuracy: vec![epoch_accuracy], dev_loss: None, dev_accuracy: None, time: vec![timer.elapsed().as_secs() as i64]
                };

                // add dev set calculation, update and early break
                if devset_iter.is_some() {

                    let dev_iter = devset_iter.as_mut().unwrap();
                    let (dev_loss, dev_accuracy) = self.validate(dev_iter, model, clip_norm);
                    progress_entry.dev_loss = Some(vec![dev_loss]);
                    progress_entry.dev_accuracy = Some(vec![dev_accuracy]);

                    if self.break_early(epoch_loss, dev_loss) {
                        break;
                    }

                }

                // print progress
                println!("{}", progress_entry);
                train_progress = train_progress.add(progress_entry);

            }

            // save model?

            Ok(())

        
        }

        fn step(&self, xs: Tensor, ys: Tensor, model: &impl ModuleT, opt: Option<&mut Optimizer>, loss: &mut f64, accuracy: &mut f64, clip_norm: f64) {

            let train_mode = match &opt {
                Some(_) => true,
                None => false
            };

            let logits = model.forward_t(&xs, train_mode); // move throught model...
            // logits of shape (sequence_length, token_vocab_size)

            let timer = Instant::now();
            let batch_loss = logits.cross_entropy_for_logits(&ys);
            println!("cross entropy part : {}", timer.elapsed().as_nanos());

            let timer = Instant::now();
            if train_mode {
                println!("batch loss: {:#?}", batch_loss);
                opt.unwrap().backward_step_clip(&batch_loss, clip_norm);
            }
            println!("opt part : {}", timer.elapsed().as_nanos());

            let timer = Instant::now();
            *loss += f64::try_from(batch_loss.mean(Kind::Float)).unwrap();
            *accuracy += self.predict(&ys, &logits);
            println!("predict part : {}", timer.elapsed().as_nanos());

        }

        fn validate(&self, devset_iter: &mut Loader, model: &impl ModuleT, clip_norm: f64) -> (f64, f64) {

            let mut total = 0.0;
            let mut loss = 0.0;
            let mut accuracy = 0.0;

            for (xs, ys) in devset_iter.shuffle().to_stream().into_iter() {

                // already in device
                // xs of shape (sequence_length, max_token_length)
                // ys of shape (sequence_length)                
                self.step(xs, ys, model, None, &mut loss, &mut accuracy, clip_norm);
                total += 1.0;
            }

            (loss / total, accuracy / total)

        }

        fn predict(&self, targets: &Tensor, logits: &Tensor) -> f64 {

            // targets are of shape (batch_size * sequence_length)
            // logits are of shape (batch_size * sequence_length, vocab_size)

            // create predictions from logits based on argmax
            let predictions = logits.argmax(1, false);
            let compare = predictions.eq_tensor(targets);
            let accuracy = compare.mean(Kind::Float).double_value(&[]);
            accuracy
        }

        fn init_optimizer(&self, vars: &VarStore, learning_rate: f64) -> Optimizer {

            let opt: Optimizer = Adam::default().build(&vars, learning_rate).unwrap();
            opt

        }

    }


    #[derive(Debug)]
    struct TrainingProgress {
        epoch: Vec<i64>,
        epoch_loss: Vec<f64>,
        epoch_accuracy: Vec<f64>,
        dev_loss: Option<Vec<f64>>,
        dev_accuracy: Option<Vec<f64>>,
        time: Vec<i64>
    }

    impl TrainingProgress {
        fn init_with_dev() -> Self {
            Self {
                epoch: vec![],
                epoch_loss: vec![],
                epoch_accuracy: vec![],
                dev_loss: Some(vec![]),
                dev_accuracy: Some(vec![]),
                time: vec![]
            }
        }
        fn init_no_dev() -> Self {
            Self {
                epoch: vec![],
                epoch_loss: vec![],
                epoch_accuracy: vec![],
                dev_loss: None,
                dev_accuracy: None,
                time: vec![]
            }
        }
    }

    impl Add for TrainingProgress {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {

            let mut new_epoch = self.epoch;
            new_epoch.extend(rhs.epoch);

            let mut new_epoch_loss = self.epoch_loss;
            new_epoch_loss.extend(rhs.epoch_loss);

            let mut new_epoch_accuracy = self.epoch_accuracy;
            new_epoch_accuracy.extend(rhs.epoch_accuracy);

            let mut new_dev_loss = None;
            if self.dev_loss.is_some() {
                new_dev_loss = self.dev_loss;
                let add_dev_loss = rhs.dev_loss.unwrap_or(vec![]);
                new_dev_loss.clone().unwrap().extend(add_dev_loss);
            }

            let mut new_dev_accuracy = None;
            if self.dev_accuracy.is_some() {
                new_dev_accuracy = self.dev_accuracy;
                let add_dev_accuracy = rhs.dev_accuracy.unwrap_or(vec![]);
                new_dev_accuracy.clone().unwrap().extend(add_dev_accuracy);
            }

            let mut new_time = self.time;
            new_time.extend(rhs.time);

            TrainingProgress {
                epoch: new_epoch,
                epoch_loss: new_epoch_loss,
                epoch_accuracy: new_epoch_accuracy,
                dev_loss: new_dev_loss,
                dev_accuracy: new_dev_accuracy,
                time: new_time
            }


        }
    }

    impl Display for TrainingProgress {

        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            
            let n = self.epoch.len();
            assert!(n > 0);
            
            let epoch = self.epoch.get(n-1).unwrap();
            let epoch_loss = self.epoch_loss.get(n-1).unwrap();
            let epoch_acc = self.epoch_accuracy.get(n-1).unwrap();
            let time = self.time.get(n-1).unwrap();

            let mut to_print = format!("epoch: {}, time (train): {}, train loss: {}, train acc: {}, ", epoch, time, epoch_loss, epoch_acc);

            if let Some(dev_loss) = &self.dev_loss {
                to_print += &format!("dev loss: {}, ", dev_loss.get(n-1).unwrap());
            }

            if let Some(dev_accuracy) = &self.dev_accuracy {
                to_print += &format!("dev acc: {}", dev_accuracy.get(n-1).unwrap());
            }

            write!(f, "{}", to_print)

        }
    }



}
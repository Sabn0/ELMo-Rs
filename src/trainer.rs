
pub mod training {

    use std::error::Error;
    use std::fmt::Display;
    use std::ops::Add;
    use std::time::Instant;
    use tch::{Tensor, Kind};
    use tch::nn::{VarStore, ModuleT, Optimizer, Adam, OptimizerConfig};
    use crate::config::JsonELMo;
    use crate::{ELMo, Loader};

    pub trait TrainModel {
        
        // train forces (x,y) labels (classification)
        fn train(&self, trainset_iter: &mut Loader, devset_iter: &mut Option<Loader>, learning_rate: f64, max_iter: i64, model: &impl ModuleT, vars: &mut VarStore, clip_norm: f64, save_model: &str, to_break_early: bool) -> Result<(), Box<dyn Error>>;
        fn validate(&self, devset_iter: &mut Loader, model: &impl ModuleT) -> (f64, f64);
        fn step(&self, xs: Tensor, ys: Tensor, model: &impl ModuleT, loss: &mut f64, accuracy: &mut f64, opt_vars: Option<(&mut Optimizer, f64)>);       
        fn predict(&self, targets: &Tensor, logits: &Tensor) -> f64;
        fn init_optimizer(&self, vars: &VarStore, learning_rate: f64) -> Result<Optimizer, Box<dyn Error>>;
        fn break_early(&self, _train_progress: &TrainingProgress) -> bool { false }
        fn save_model(&self, out_path: &str, vars: &VarStore) -> Result<(), Box<dyn Error>> { Ok(vars.save(out_path)?) }
    }

    pub struct ElmoTrainer;

    impl ElmoTrainer {

        pub fn new() -> Self {
             Self {} 
        }

        pub fn run_training(&self, trainset_iter: &mut Loader, devset_iter: &mut Option<Loader>, model: &ELMo, vars: &mut VarStore, params: &JsonELMo) -> Result<(), Box<dyn Error>> {

            self.train(trainset_iter, devset_iter, params.learning_rate, params.max_iter, model, vars, params.clip_norm, &params.output_file, params.break_early)?;
            Ok(())
        }

        pub fn run_testing(&self, testset_iter: &mut Loader, model: &ELMo) -> Result<f64, Box<dyn Error>> {
            let (_, acc) = self.validate(testset_iter, model);
            Ok(acc)
        }

    }

    impl TrainModel for ElmoTrainer {
        
        fn train(&self, trainset_iter: &mut Loader, devset_iter: &mut Option<Loader>, learning_rate: f64, max_iter: i64, model: &impl ModuleT, vars: &mut VarStore, clip_norm: f64, output_file: &str, to_break_early: bool) -> Result<(), Box<dyn Error>> {
            
            let mut opt = self.init_optimizer(&vars, learning_rate)?;
            let mut train_progress = match devset_iter {
                Some(_) => TrainingProgress::init_with_dev(),
                None => TrainingProgress::init_no_dev()
            };
            
            for epoch in 0..max_iter {

                let timer = Instant::now();
                let mut total = 0.0;
                let mut epoch_loss = 0.0;
                let mut epoch_accuracy = 0.0;
                let batch_size = (&trainset_iter).batch_size;

                for (xs, ys) in trainset_iter.shuffle().to_stream().into_iter() {

                    // xs of shape (batch_size, seq_length, max_token_length)
                    // ys of shape (batch_size, seq_length)
                    self.step(xs, ys, model, &mut epoch_loss, &mut epoch_accuracy, Some((&mut opt, clip_norm)));
                    total += batch_size as f64;
                }

                // update training progress
                epoch_loss /= total;
                epoch_accuracy /= total;

                let mut progress_entry = TrainingProgress {
                    epoch: vec![epoch], 
                    epoch_loss: vec![epoch_loss], 
                    epoch_accuracy: vec![epoch_accuracy], 
                    dev_loss: None, 
                    dev_accuracy: None, 
                    time: vec![timer.elapsed().as_secs() as i64]
                };

                // add dev set calculation, update and early break
                if devset_iter.is_some() {

                    let dev_iter = devset_iter.as_mut().unwrap();
                    let (dev_loss, dev_accuracy) = self.validate(dev_iter, model);
                    progress_entry.dev_loss = Some(vec![dev_loss]);
                    progress_entry.dev_accuracy = Some(vec![dev_accuracy]);

                    if to_break_early && self.break_early(&train_progress) {
                        break;
                    }

                }

                // print progress
                train_progress = train_progress.add(progress_entry);
                println!("{}", train_progress);

            }

            self.save_model(output_file, vars)?;

            Ok(())

        
        }

        fn step(&self, xs: Tensor, ys: Tensor, model: &impl ModuleT, loss: &mut f64, accuracy: &mut f64, opt_vars: Option<(&mut Optimizer, f64)>) {
            
            let train_mode = match &opt_vars {
                Some(_) => true,
                None => false
            };

            let logits = model.forward_t(&xs, train_mode); // move throught model...
            // logits of shape (batch_size * seq_length, token_vocab_size), match the targets to that shape
            let targets = ys.reshape(&[-1]);
            let batch_loss = logits.cross_entropy_for_logits(&targets);
            if train_mode {
                let opt_vars = opt_vars.unwrap();
                let opt = opt_vars.0;
                let _clip_norm = opt_vars.1;
                opt.backward_step(&batch_loss);
                //opt.backward_step_clip(&batch_loss, _clip_norm);
            }

            *loss += f64::try_from(batch_loss.mean(Kind::Float)).unwrap();
            *accuracy += self.predict(&targets, &logits);
        }

        fn validate(&self, devset_iter: &mut Loader, model: &impl ModuleT) -> (f64, f64) {

            let mut total = 0.0;
            let mut loss = 0.0;
            let mut accuracy = 0.0;
            let batch_size = (&devset_iter).batch_size;

            for (xs, ys) in devset_iter.shuffle().to_stream().into_iter() {

                // already in device
                // xs of shape (sequence_length, max_token_length)
                // ys of shape (sequence_length)                
                self.step(xs, ys, model, &mut loss, &mut accuracy, None);
                total += batch_size as f64;
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

        fn init_optimizer(&self, vars: &VarStore, learning_rate: f64) -> Result<Optimizer, Box<dyn Error>> {

            let opt: Optimizer = Adam::default().build(&vars, learning_rate)?;
            Ok(opt)

        }

        fn break_early(&self, train_progress: &TrainingProgress) -> bool {

            let epochs = &train_progress.epoch;
            let n = epochs.len();
            if n <= 2 {
                return false
            }

            let diff_train_loss = train_progress.epoch_loss.get(n-1).unwrap() - train_progress.epoch_loss.get(n-2).unwrap();

            while let Some(dev_loss) = &train_progress.dev_loss {
                let diff_dev_loss = dev_loss.get(n-1).unwrap() - dev_loss.get(n-2).unwrap();
                if diff_train_loss < 0.0 && diff_dev_loss > 0.0 {
                    return true;
                } else {
                    return false;
                }
            }

            false
        }

    }


    #[derive(Debug)]
    pub struct TrainingProgress {
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
                let mut prior_dev_loss = self.dev_loss.unwrap();
                let add_dev_loss = rhs.dev_loss.unwrap_or(vec![]);
                prior_dev_loss.extend(add_dev_loss);
                new_dev_loss = Some(prior_dev_loss);
            }

            let mut new_dev_accuracy = None;
            if self.dev_accuracy.is_some() {
                let mut prioer_dev_accuracy = self.dev_accuracy.unwrap();
                let add_dev_accuracy = rhs.dev_accuracy.unwrap_or(vec![]);
                prioer_dev_accuracy.extend(add_dev_accuracy);
                new_dev_accuracy = Some(prioer_dev_accuracy);
            }

            let mut new_time = self.time;
            new_time.extend(rhs.time);

            let new_training_progress = TrainingProgress {
                epoch: new_epoch,
                epoch_loss: new_epoch_loss,
                epoch_accuracy: new_epoch_accuracy,
                dev_loss: new_dev_loss,
                dev_accuracy: new_dev_accuracy,
                time: new_time
            };

            new_training_progress

        }
    }

    impl Display for TrainingProgress {

        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            
            let n = self.epoch.len();
            assert!(n > 0, "can't use display before first iteration");
            
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
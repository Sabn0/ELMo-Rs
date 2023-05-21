


use std::env;

use elmo_trainer::ConfigElmo;
use elmo_trainer::ELMoText;
use elmo_trainer::Loader;
use elmo_trainer::Splitter;
use elmo_trainer::files_handling;
use elmo_trainer::Preprocessor;
use elmo_trainer::DatasetBuilder;
use elmo_trainer::training::ElmoTrainer;
use elmo_trainer::ELMo;
use tch::Tensor;
use tch::nn;


fn main() {

    //
    // loading training parameteres
    println!("entering program...");
    let args: Vec<String> = env::args().collect();

    println!("building parameters...");
    let mut params = match ConfigElmo::new(&args) {
        Ok(config) => config.get_params(),
        Err(e) => panic!("{}", e)
    };
    println!("{}", params);
    // -- end of loading parameters --
    //

    //
    // preprocess of sentences
    let mut sentences = files_handling::load_sentences(&params.corpus_file).unwrap();
    let mut preprocessor = Preprocessor::new();
    let (token2int,char2int) = preprocessor.preprocess(
        &mut sentences,
        &mut params.token_vocab_size,
        &mut params.char_vocab_size,
        params.min_count, 
        params.char_start, 
        params.char_end,
        params.char_unk,
        params.str_unk.as_str()
    );
    // -- end of preprocessing sentences
    //

    //
    // Create an ELM0 textual loader - data builder that moves data from strings to ints
    let n_samples = (&sentences).len() as i64;
    let elmo_text_loader = ELMoText::new(
        sentences, 
        token2int, 
        char2int, 
        Some(params.max_len_token as usize), 
        params.char_start, 
        params.char_end,
        params.char_unk,
        params.str_unk.to_string()
    );
    // -- end of data building --
    //

    //
    // Create an instance of the ELMo model
    let mut vars = nn::VarStore::new(params.device);
    let model = ELMo::new(
        &vars.root(),
        params.n_lstm_layers,
        params.in_dim,
        params.hidden_dim,
        params.char_vocab_size,
        params.token_vocab_size, 
        params.char_embedding_dim, 
        params.in_channels, 
        params.out_channels, 
        params.kernel_size, 
        params.highways, 
        params.dropout
    );
    // -- end of instantiating model --
    //

    //
    // spliting data to train, dev and test sets, and moving to loaders (iterators over examples)
    let splitter = Splitter::new();
    let splits: Vec<Tensor> = splitter.get_split_train_dev_test_indices(n_samples);
    let mut iters = splits.iter().map(|split| {
        
        let indices = TryInto::<Vec<i64>>::try_into(split).unwrap();
        let (xs, ys): (Vec<_>, Vec<_>) = indices.iter()
        .map(|i| elmo_text_loader.get_example(*i as usize).unwrap())
        .unzip();

        Loader::new(xs, ys, params.device, params.batch_size, params.seq_length)
    });
    // -- end of creating train, dev, test iterators
    //
    
    //
    // running the training process with train and dev iterators
    let mut trainset_iter = iters.next().unwrap();
    let mut devset_iter = iters.next();
    let elmo_train = ElmoTrainer::new();
    if let Err(e) = elmo_train.run_training(
                &mut trainset_iter, 
                &mut devset_iter, 
                params.learning_rate, 
                params.max_iter, 
                model,
                &mut vars,
                params.clip_norm,
                params.token_vocab_size
    ) {
        panic!("{}", e)
    };
    // -- end of training process --
    //

    // 
    // do testing on test set with saved model

    // -- end of testing --
    //
}   
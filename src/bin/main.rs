


use std::env;

use elmo_trainer::ConfigElmo;
use elmo_trainer::ELMoText;
use elmo_trainer::Loader;
use elmo_trainer::Splitter;
use elmo_trainer::files_handling;
use elmo_trainer::Preprocessor;
use elmo_trainer::DatasetBuilder;
use elmo_trainer::training::ElmoTrainer;
use tch::Tensor;
use tch::nn;
use elmo_trainer::ELMo;

fn main() {

    // since this is a 1 batch, maybe removing it will be faster ...

    println!("entering program...");
    let _args: Vec<String> = env::args().collect();

    let args = vec!["".to_string(), "args.json".to_string()];
    println!("building parameters...");
    let mut params = match ConfigElmo::new(&args) {
        Ok(config) => config.get_params(),
        Err(e) => panic!("{}", e)
    };
    println!("{}", params);

    let mut sentences = files_handling::load_sentences(&params.corpus_file).unwrap();
    let mut preprocessor = Preprocessor::new();
    let (token2int,char2int) = preprocessor.preprocess(&mut sentences,
        params.token_vocab_size as usize,
        params.min_count as usize, 
        params.char_start, 
        params.char_end,
        params.char_unk,
        &params.str_unk
    );

    // update vocabs
    params.token_vocab_size = token2int.len() as i64;
    println!("working on token vocab : {}", params.token_vocab_size);
    params.char_vocab_size = char2int.len() as i64;
    println!("working on char vocab : {}", params.char_vocab_size);

    let n_samples = (&sentences).len() as i64;
    let elmo_text_loader = ELMoText::new(sentences, 
        token2int, 
        char2int, 
        Some(params.max_len_token as usize), 
        params.char_start, 
        params.char_end,
        params.char_unk,
        params.str_unk.to_string());

    let splitter = Splitter::new();
    let splits: Vec<Tensor> = splitter.get_split_train_dev_test_indices(n_samples);

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

    let mut iters = splits.iter().map(|split| {
        
        let indices = TryInto::<Vec<i8>>::try_into(split).unwrap();
        let (xs, ys): (Vec<_>, Vec<_>) = indices.iter()
        .map(|i| elmo_text_loader.get_example(*i as usize).unwrap())
        .unzip();

        Loader::new(xs, ys, params.device)
    });
    
    let mut trainset_iter = iters.next().unwrap();
    let mut devset_iter = iters.next();

    let elmo_train = ElmoTrainer::new();
    if let Err(e) = elmo_train.run_training(
                &mut trainset_iter, 
                &mut devset_iter, 
                params.learning_rate, 
                params.max_iter, 
                model, params.token_vocab_size, 
                &mut vars) 
    {
        panic!("{}", e);
    };
    

}   
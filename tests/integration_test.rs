
use elmo_trainer::{Preprocessor, ELMoText, JsonELMo, ELMo, Splitter, training::ElmoTrainer, DatasetBuilder, Loader};
use tch::{Device, nn, Tensor};


#[test]
fn integration_without_configure() {

    // example sentences (10 for validation)
    let mut sentences = [
        "This is a first sentence",
        "This is a second sentence",
        "This is a third sentence",
        "This is just another sentence",
        "Those are configuration inputs here",
        "A B C D E",
        "1 2 3 4 5",
        "a b c d e",
        "Blue sky , yellow sun",
        "sky is blue not yellow"
    ].map(|x| x.to_string()).to_vec();

    // example parameters
    let mut params = JsonELMo { 
        corpus_file: None, 
        output_file: None,
        token_vocab_size: 50, // the examples have less
        char_vocab_size: 50, // the examples have less
        min_count: 1,
        max_len_token: 20,
        char_start: '$',
        char_end: '^',
        char_unk: '~',
        str_unk: String::from("UNK"),
        batch_size: 1,
        seq_length: 1,
        char_embedding_dim: 5,
        in_channels: 1,
        out_channels: vec![20],
        kernel_size: vec![1],
        highways: 1, 
        in_dim: 10, 
        hidden_dim: 10,
        n_lstm_layers: 1, 
        dropout: 0.0,
        device: Device::cuda_if_available(),
        max_iter: 2, 
        learning_rate: 0.1, 
        clip_norm: 0.0, 
        break_early: false 
    };

    //
    // preprocess of sentences
    let mut preprocessor = Preprocessor::new();
    let (token2int,char2int) = preprocessor.preprocess(&mut sentences, &mut params);
    assert_eq!(token2int.len(), 38); // self counted, with spacial tokens
    // -- end of preprocessing sentences
    //

    //
    // Create an ELMo textual loader - data builder that moves data from strings to ints
    let n_samples = (&sentences).len() as i64;
    assert_eq!(n_samples, 10);
    let elmo_text_loader = ELMoText::new(sentences, token2int, char2int, &params);
    // -- end of data building --
    //

    //
    // Create an instance of the ELMo model
    let mut vars = nn::VarStore::new(params.device);
    let model = ELMo::new(&vars.root(), &params);
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
    let mut trainset_iter = iters.next().ok_or("iters is but should have multiple loaders empty").unwrap();
    let mut devset_iter = iters.next();
    let elmo_train = ElmoTrainer::new();
    let res = elmo_train.run_training(&mut trainset_iter, &mut devset_iter, &model, &mut vars, &params);
    assert!(res.is_ok());
    // -- end of training process --
    //


}



use tch::Device;
use tch::Kind;
use tch::Tensor;
use tch::nn;
use tch::data::Iter2;
use elmo_trainer::ELMo;
use tch::nn::Adam;
use tch::nn::ModuleT;
use tch::nn::OptimizerConfig;

fn main() {

    let x: Vec<String> = Vec::new();
    let _n_samples = x.len() as i64;
    let vocab_size = 2;
    let char_vocab_size = 5;
    let _min_count = 1;
    let max_len_token = 7;
    let _char_start = '$';
    let _char_end = '^';
    let _str_unk = "UNK";
    let batch_size = 1;
    let char_embedding_dim = 100;
    let in_channels = char_embedding_dim;
    let out_channels = vec![10,20,30,40];
    let kernel_size = vec![2,3,4,5];
    let highways = 2;
    let in_dim = 512;
    let hidden_size = 4096;
    let lstm_layers = 2;

    let device = Device::cuda_if_available();
    println!("{:?}", &device);
    let vars = nn::VarStore::new(device);
    let model = ELMo::new(
        &vars.root(),
        lstm_layers,
        in_dim,
        hidden_size,
        char_vocab_size,
        vocab_size, 
        char_embedding_dim, 
        in_channels, 
        out_channels, 
        kernel_size, 
        highways);

    let xs = Tensor::ones(&[3, 9, max_len_token], (Kind::Int, Device::Cpu));
    let ys = Tensor::ones(&[3, 9, max_len_token], (Kind::Int, Device::Cpu));
    let mut opt = Adam::default().build(&vars, 1e-4).unwrap();
    
    let mut iter = Iter2::new(&xs, &ys, batch_size);

    for (x, y) in iter.shuffle().into_iter().to_device(vars.device()) {
        // move throught training...
        let out = model.forward_t(&x, true);
        // out and y should be both of shape (batch_size, sequence_length, vocab_size)
        let loss = out.cross_entropy_for_logits(&y);

        opt.backward_step(&loss);

        let _acc = model.batch_accuracy_for_logits(&out, &y, vars.device(), batch_size);
    }

    /* 
    let mut preprocessor = do_preprocess::Preprocessor::new();
    let (token2int,char2int) = preprocessor.preprocess(&mut x, vocab_size, min_count, char_start, char_end, str_unk);

    let elmo_text_loader = ELMoText::new(x, token2int, char2int, Some(max_len_token), char_start, char_end, str_unk.to_string());

    let loader = Loader::new();
    let splits: Vec<Tensor> = loader.get_split_train_dev_test_indices(n_samples);
    for split in splits {

        let indices = TryInto::<Vec<i8>>::try_into(split).unwrap();
        let (xs, ys): (Vec<_>, Vec<_>) = indices.iter()
        .map(|i| elmo_text_loader.get_example(*i as usize).unwrap())
        .unzip();
        let xs = Tensor::concat(&xs, 0);
        let ys = Tensor::concat(&ys, 0);

        let mut iter = Iter2::new(&xs, &ys, batch_size);

        for (x, y) in iter.shuffle().into_iter() {
            // move throught training...
        }

    }
    */



}   
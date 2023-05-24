
// imports
use std::iter::zip;
use std::ops::Mul;
use tch::{nn, Tensor, IndexOp, Device};
use tch::nn::{ModuleT, RNN};
use crate::config::JsonELMo;

// an self-implementation of biLSTM and a char-level 
// convolution as described in the ELMo paper https://aclanthology.org/N18-1202.pdf

#[derive(Debug)]
pub(in self) struct CnnBlock {
    conv: nn::Conv<[i64; 2]>,
    kernel_size: i64
}

impl CnnBlock {
    fn new(vars: &nn::Path, in_channels: i64, out_channels: i64, kernel_size: i64, embedding_dim: i64) -> Self {

        // If a word is of length k charachters, the convolution is What's described as
        // a narrow convolution between a charachter within a word, C_k of shape (d, l),  
        // and a kernel H of shape (d, w), where d is char embedding, l is the length of the word,
        // and w is the size of the kernal. The multiplication is done as a dot product.
        // number of filters is handled as out_channels

        let conv = nn::conv(vars / "conv", in_channels, out_channels, [embedding_dim, kernel_size], Default::default());
        let kernel_size = kernel_size;
        
        Self {
            conv: conv,
            kernel_size: kernel_size,
        }

    }
}

impl ModuleT for CnnBlock {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        
        // xs is of shape (batch_size, 1, token_length, embedding_dim)
        let dims = xs.internal_shape_as_tensor();
        let dims = Vec::<i64>::try_from(dims).unwrap();
        assert!(4 == dims.len());
        let batch_size = dims[0];
        let token_length = dims[2];
        let embedding_dim = dims[3];
        let pool_kernel: i64 = token_length - self.kernel_size + 1;

        // xs reshape : (batch_size, 1, token_length, embedding_dim) => (1, embedding_dim, token_length)
        let reshaped_xs = xs.reshape(&[batch_size, 1, embedding_dim, token_length]);

        // denotations:
        // token_length = l, 
        // out_channels = h = number of filters, 
        // kernel_size = w, 
        // embedding_dim = d
        
        // self.conv.w is of shape (h, 1, d, w)
        // conv does: (h, 1, d, w) * (batch_size, 1, d, l) => (batch_size, h, 1, l-w+1)
        let conv_out = reshaped_xs.apply(&self.conv);
        
        // tanh doesn't change dims, (batch_size, h, 1, l-w+1), then squeeze to (batch_size, h, l-w+1)
        assert!(Vec::<i64>::try_from(conv_out.internal_shape_as_tensor()).unwrap()[2] == 1);
        let act_out = conv_out.tanh().squeeze_dim(2);
         
         // max_pool1d moves xs : (batch_size, h, l-w+1) => (batch_size, h, 1)
        let pool_out = act_out.max_pool1d(&[pool_kernel], &[1], &[0], &[1], false);
        
        // the output should be (batch_size, h) after squeeze
        assert!(Vec::<i64>::try_from(pool_out.internal_shape_as_tensor()).unwrap()[2] == 1);
        let out = pool_out.squeeze_dim(2);

        out
    }
}

#[derive(Debug)]
pub(in self) struct Highway {
    w_t: nn::Linear,
    w_h: nn::Linear,
}

impl Highway {
    
    fn new(vars: &nn::Path, in_dim: i64, out_dim: i64) -> Self {

        let w_t = nn::linear(vars / "w_t", in_dim, out_dim, Default::default());
        let w_h = nn::linear(vars / "w_h", in_dim, out_dim, Default::default());
    
        Self {
            w_t: w_t,
            w_h: w_h
        }
    }

}

impl ModuleT for Highway {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        
        let t = xs.apply(&self.w_t).sigmoid();
        let transform_part = xs.apply(&self.w_h).relu().mul(&t);
        let carry_part = xs.mul(1-t);
        let out: Tensor = transform_part + carry_part;
        out
        
        // xs remains (batch_size, total_filters) from input to end, dims don't change
    }
}

#[derive(Debug)]
pub(in self) struct CharLevelNet {
    embedding: nn::Embedding,
    conv_blocks: Vec<CnnBlock>,
    highways: Vec<Highway>,
    out_linear: nn::Linear,
    device: Device
}

impl CharLevelNet {
    fn new(vars: &nn::Path,
         vocab_size: i64, 
         embedding_dim: i64, 
         in_channels: i64, 
         out_channels: Vec<i64>, 
         kernel_size: Vec<i64>, 
         highways: i64, 
         char_level_out_dim: i64) -> Self {


        // creation of M convolution blocks based M kernel sizes and M out channels
        let embedding = nn::embedding(vars / "embed", vocab_size, embedding_dim, Default::default());
        let mut conv_blocks = Vec::new();
        for (out_channel, kernel_size) in zip(&out_channels, kernel_size) {
            let conv_block = CnnBlock::new(vars, in_channels, *out_channel, kernel_size, embedding_dim);
            conv_blocks.push(conv_block);
        }

        // total filters should be the sum over out_channels
        let total_filters: i64 = (&out_channels).iter().sum();

        // creation of N highways
        let mut highway_layers = Vec::new();
        for _ in 0..highways {
            let highway = Highway::new(vars, total_filters, total_filters);
            highway_layers.push(highway);
        }

        // move to some representation dimenstion
        let out_linear = nn::linear(vars / "to_dim", total_filters, char_level_out_dim, Default::default());
        
        Self {
            embedding: embedding,
            conv_blocks: conv_blocks,
            highways: highway_layers,
            out_linear: out_linear,
            device: vars.device()
        }

    }
    

}

impl ModuleT for CharLevelNet {
    
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {

        // xs is of shape (batch_size, seq_length, token_length)
        let dims = xs.internal_shape_as_tensor();
        let dims = Vec::<i64>::try_from(dims).unwrap();
        let seq_length = &dims[1];
        
        // iterate over tokens, do convolution to each token at a time
        let mut outputs = Vec::new();
        for s in 0..*seq_length {

            let xs_tokens: Tensor = xs.slice(1, s, s+1, 1); // should be (batch_size, 1, token_length)
            let xs_embedded = xs_tokens.apply(&self.embedding); // should be (batch_size, 1, token_length, embedding_dim)
            let mut token_outputs = Vec::new();
            for conv_block in &self.conv_blocks {
                let out = conv_block.forward_t(&xs_embedded, train); // out is of shape (batch_size, n_filters)
                token_outputs.push(out);
            }

            // each output in token_outputs is of shape n_kernels * (batch_size, n_filters,) => (batch_size, total_filters)
            let mut flatten_token_outputs = Tensor::concat(&token_outputs, 1).to_device(self.device);

            // move through highways, remains (batch_size, total_filters)
            for highway in &self.highways {
                flatten_token_outputs = highway.forward_t(&flatten_token_outputs, train);
            }
            outputs.push(flatten_token_outputs);
        }

        // seq_length * (batch_size, total_filters) => (batch_size, seq_length, total_filters)
        let outs = Tensor::stack(&outputs, 1).to_device(self.device);

        // move to linear out (batch_size, seq_length, total_filters) => (batch_size, seq_length, out_linear)
        let out = outs.apply(&self.out_linear);
        out


    }
}


#[derive(Debug)]
pub(in self) struct UniLM {
    lstm_layers: Vec<nn::LSTM>,
    to_rep: nn::Linear,
    dropout: f64,
    device: Device
}

impl UniLM {
    fn new(vars: &nn::Path, n_lstm_layers: i64, in_dim: i64, hidden_dim: i64, dropout: f64) -> Self {

        // creation of N unidirectional lstm layers
        let mut lstm_layers = Vec::new();
        for _ in 0..n_lstm_layers {

            // default on rnn gives everything we need except for dropout, taken care in forward
            let lm = nn::lstm(vars / "lstm", in_dim, hidden_dim, Default::default());
            lstm_layers.push(lm);
        }

        // move to some representaion layer
        let to_rep = nn::linear(vars / "to_dim_lstm", hidden_dim, in_dim, Default::default());

        Self {
            lstm_layers: lstm_layers,
            to_rep: to_rep,
            dropout: dropout,
            device: vars.device()
        }


    }
}

impl ModuleT for UniLM {

    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        
        // xs should be (batch_size, seq_length, out_linear)

        // need residual connections, so lstm out should be the same size of input
        let mut out_point = xs.to_owned().shallow_clone().to_device(self.device);
        let mut outputs = vec![xs.to_owned().shallow_clone().to_device(self.device)];

        for (j, lstm) in (&self.lstm_layers).iter().enumerate() {

            outputs.push(out_point.shallow_clone().to_device(self.device));

            // adding dropout at non-test time
            let out_lstm = lstm.seq(&out_point.dropout(self.dropout, train).to_device(self.device));
            out_point = out_lstm.0;
            
            // out moves back to shape (batch_size, seq_length, hidden_dim) => (batch_size, seq_length, out_linear)
            out_point = out_point.apply(&self.to_rep);

            // adding residual to out
            out_point += outputs[j].shallow_clone().to_device(self.device);

        }

        // move n_lstm_layers * (batch_size, seq_length, out_linear) =>  (n_lstm_layers, batch_size, seq_length, out_linear)
        let out = Tensor::stack(&outputs, 0).to_device(self.device);
        out

    }
}

#[derive(Debug)]
pub struct ELMo {
    forward_lm: UniLM,
    backward_lm: UniLM,
    to_vocab: nn::Linear,
    n_lstm_layers: i64,
    char_level: CharLevelNet,
    token_vocab_size: i64,
    device: Device
}

impl ELMo {
    pub fn new(vars: &nn::Path, params: &JsonELMo
    ) -> Self {

        // unroll parameters
        let n_lstm_layers = params.n_lstm_layers;
        let in_dim = params.in_dim;
        let hidden_dim = params.hidden_dim;
        let char_vocab_size= params.char_vocab_size;
        let token_vocab_size = params.token_vocab_size;
        let char_embedding_dim = params.char_embedding_dim;
        let in_channels = params.in_channels;
        let out_channels = params.out_channels.clone();
        let kernel_size = params.kernel_size.clone();
        let highways = params.highways;
        let dropout = params.dropout;
        
        let char_level = CharLevelNet::new(vars, char_vocab_size, char_embedding_dim, in_channels, out_channels, kernel_size, highways, in_dim);
        let forward_lm = UniLM::new(vars, n_lstm_layers, in_dim, hidden_dim, dropout);
        let backward_lm = UniLM::new(vars, n_lstm_layers, in_dim, hidden_dim, dropout);
        let to_vocab = nn::linear(vars / "to_vocab", in_dim, token_vocab_size, Default::default());

        Self {
            forward_lm: forward_lm,
            backward_lm: backward_lm,
            to_vocab: to_vocab,
            n_lstm_layers: n_lstm_layers,
            char_level: char_level,
            token_vocab_size: token_vocab_size,
            device: vars.device()
        }


    }
}

impl ModuleT for ELMo {

    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        
        // xs is of shape (batch_size, seq_length, token_length)
        // move through char enconding => (batch_size, seq_length, out_linear)
        let xs_embedded = &self.char_level.forward_t(xs, train);

        // xs_embedded should be (batch_size, seq_length, out_linear)
        let xs_embedded_flip = Tensor::flip(&xs_embedded.to_owned().shallow_clone(), &[0]).to_device(self.device);

        // both should be (n_lstm_layers, batch_size, seq_length, out_linear)
        let forward_lm_outs = self.forward_lm.forward_t(xs_embedded, train);
        let backward_lm_outs = self.backward_lm.forward_t(&xs_embedded_flip, train);

        // the elmo representation is a combination of all the outputs (2L) + xs
        // for the simple case, I take the sum of xs and the two last outputs
        let forward_last = forward_lm_outs.i(self.n_lstm_layers-1..self.n_lstm_layers).squeeze_dim(0);
        let backward_last = backward_lm_outs.i(self.n_lstm_layers-1..self.n_lstm_layers).squeeze_dim(0);

        // compute output representation as a mix of 3 representations
        let weights = [0.2, 0.4, 0.4];
        let out: Tensor = weights[0] * xs_embedded + weights[1] * forward_last + weights[2] * backward_last; 

        // The representation transfers to vocabulary size, (batch_size, seq_length, out_linear) => (batch_size, seq_length, token_vocab_size)
        // then also unify two first dims for loss computation
        let logits = out.apply(&self.to_vocab).reshape(&[-1, self.token_vocab_size]);
        logits
        
    }
}
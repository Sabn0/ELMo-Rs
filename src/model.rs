
use std::iter::zip;
use std::ops::Mul;

use tch::{nn, Tensor};
use tch::nn::{ModuleT, RNN};


// If a word is of length k charachters, the convolution is What's described as
// a narrow convolution between a charachter within a word, C_k \in (d, l),  
// and a kernel H \in (d, w), where d is char embedding, l is the length of the word,
// and w is the size of the kernal. The multiplication is done as a dot product.
#[derive(Debug)]
pub struct CnnBlock {
    conv: nn::Conv1D,
    kernel_size: i64
}

impl CnnBlock {
    fn new(vars: &nn::Path, in_channels: i64, out_channels: i64, kernel_size: i64) -> Self {

        // handling embedding dim d as input channels.
        // hadnling number of filters h as output channels.
        let conv = nn::conv1d(vars, in_channels, out_channels, kernel_size, Default::default());
        let kernel_size = kernel_size;

        Self {
            conv: conv,
            kernel_size: kernel_size
        }

    }
}

impl ModuleT for CnnBlock {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        
        // xs is of shape (batch_size, 1, token_length, embedding_dim)
        let dims = xs.internal_shape_as_tensor();
        let dims = Vec::<i64>::from(dims);
        println!("{:?}", dims);
        assert!(4 == dims.len());

        let batch_size = dims[0];
        let token_length = dims[2];
        let embedding_dim = dims[3];
        let pool_kernel: i64 = token_length - self.kernel_size + 1;

        // xs reshape : (batch_size, token_length, embedding_dim) => (batch_size, embedding_dim, token_length)
        let xs = xs.reshape(&[batch_size, embedding_dim, token_length]);

        // denote token_length = l, out_channels = h = number of filters, kernel_size = w, embedding_dim = d
        // self.conv.w is of shape (h, d, w)
        // conv does (h,d,w) * (batch_size, d, l) => (batch_size, h, l-w+1)
        let conv_out = xs.apply(&self.conv);

        // tanh doesn't change dims, (batch_size, h, l-w+1)
        let act_out = conv_out.tanh();
         
         // max_pool1d moves xs : (batch_size, h, l-w+1) => (batch_size, h, 1)
        let pool_out = act_out.max_pool1d(&[pool_kernel], &[1], &[0], &[1], false);

        // the output should be (batch_size, h)
        let out = pool_out.squeeze_dim(2);

        
        out
    }
}

#[derive(Debug)]
struct Highway {
    w_t: nn::Linear,
    w_h: nn::Linear,
}

impl Highway {
    
    fn new(vars: &nn::Path, in_dim: i64, out_dim: i64) -> Self {

        let w_t = nn::linear(vars, in_dim, out_dim, Default::default());
        let w_h = nn::linear(vars, in_dim, out_dim, Default::default());
    
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

        // xs should remain (batch_size, total_filters) from input to end
    }
}

#[derive(Debug)]
pub struct CharLevelNet {
    embedding: nn::Embedding,
    conv_blocks: Vec<CnnBlock>,
    highways: Vec<Highway>,
    out_linear: nn::Linear
}

impl CharLevelNet {
    pub fn new(vars: &nn::Path,
         vocab_size: i64, 
         embedding_dim: i64, 
         in_channels: i64, 
         out_channels: Vec<i64>, 
         kernel_size: Vec<i64>, 
         highways: i64, 
         char_level_out_dim: i64) -> Self {

            // out_channels = number of filters
            // kernel_size = matching kernel width

        let embedding = nn::embedding(vars, vocab_size,  embedding_dim, Default::default());
        let mut conv_blocks = Vec::new();
        for (out_channel, kernel_size) in zip(&out_channels, kernel_size) {
            let conv_block = CnnBlock::new(vars, in_channels, *out_channel, kernel_size);
            conv_blocks.push(conv_block);
        }

        // total filters should be the sum over out_channels
        let total_filters: i64 = (&out_channels).iter().sum();
        let mut highway_layers = Vec::new();
        for _ in 0..highways {
            let highway = Highway::new(vars, total_filters, total_filters);
            highway_layers.push(highway);
        }

        // output to linear
        let out_linear = nn::linear(vars, total_filters, char_level_out_dim, Default::default());
        
        Self {
            embedding: embedding,
            conv_blocks: conv_blocks,
            highways: highway_layers,
            out_linear: out_linear
        }

    }
    

}

impl ModuleT for CharLevelNet {
    
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {

        // xs is of shape (batch_size, sequence_length, token_length), batch_size = 1
        let dims = xs.internal_shape_as_tensor();
        let dims = Vec::<i64>::from(dims);
        let batch_size = &dims[0];
        let seq_length = &dims[1];
        println!("{:?}", dims);
        
        // iterate over tokens
        let mut outputs = Vec::new();
        for s in 0..*seq_length {

            let xs_tokens: Tensor = xs.slice(1, s, s+1, 1); // should be (batch_size, 1, token_length)
            let x = xs_tokens.apply(&self.embedding); // should be (batch_size, 1, token_length, embedding_dim)
            let mut token_outputs = Vec::new();
            for conv_block in &self.conv_blocks {
                let out = conv_block.forward_t(&x, train); // out is of shape (batch_size, n_filters)
                token_outputs.push(out);
            }

            // each output in token_outputs is of shape k * (batch_size, n_filters) => (batch_size, total_filters)
            let mut token_outputs = Tensor::concat(&token_outputs, 1);

            // move through highways, remains (batch_size, total_filters)
            for highway in &self.highways {
                token_outputs = highway.forward_t(&token_outputs, train);
            }

            outputs.push(token_outputs);
        }

        // (sequence_length, batch_size, total_filters) => (batch_size, sequence_length, total_filters)
        let outputs = Tensor::concat(&outputs, 0).reshape(&[*batch_size, *seq_length, -1]);

        // move to linear out (batch_size, sequence_length, total_filters) => (batch_size, sequence_length, out_linear)
        let outputs = outputs.apply(&self.out_linear);
        outputs


    }
}


#[derive(Debug)]
pub struct UniLM {
    lstm_layers: Vec<nn::LSTM>,
    to_rep: nn::Linear,
    dropout: f64
}

impl UniLM {
    pub fn new(vars: &nn::Path, n_lstm_layers: i64, in_dim: i64, hidden_dim: i64, dropout: f64) -> Self {

        let mut lstm_layers = Vec::new();
        for _ in 0..n_lstm_layers {

            // default on rnn gives everything we need except for dropout, taken care in forward
            let lm = nn::lstm(vars, in_dim, hidden_dim, Default::default());
            lstm_layers.push(lm);
        }

        let to_rep = nn::linear(vars, hidden_dim, in_dim, Default::default());

        Self {
            lstm_layers: lstm_layers,
            to_rep: to_rep,
            dropout: dropout
        }


    }
}

impl ModuleT for UniLM {

    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        
        // xs should be (batch_size, sequence_length, out_linear)
        
        // need residual connections, so lstm out should be the same size of input
        let mut out = xs.to_owned().shallow_clone();
        let mut outputs = vec![xs.to_owned().shallow_clone()];

        for (j, lstm) in (&self.lstm_layers).iter().enumerate() {

            outputs.push(out.shallow_clone());

            // adding dropout at non-test time
            let out_lstm = lstm.seq(&out.dropout(self.dropout, train));
            out = out_lstm.0;
            
            // out moves back to shape (batch_size, sequence_length, hidden_dim) => (batch_size, sequence_length, out_linear)
            out = out.apply(&self.to_rep);

            // adding residual to out
            out += outputs[j].shallow_clone();

        }

        // move n_lstm_layers * (batch_size, sequence_length, out_linear) =>  (n_lstm_layers, batch_size, sequence_length, out_linear)
        let outputs = Tensor::concat(&outputs, 0);
        outputs

    }
}

#[derive(Debug)]
pub struct ELMo {
    forward_lm: UniLM,
    backward_lm: UniLM,
    to_vocab: nn::Linear,
    n_lstm_layers: i64,
    char_level: CharLevelNet
}

impl ELMo {
    pub fn new(vars: &nn::Path, 
        n_lstm_layers: i64, 
        in_dim: i64, 
        hidden_dim: i64,
        char_vocab_size: i64,
        token_vocab_size: i64,
        char_embedding_dim: i64,
        in_channels: i64,
        out_channels: Vec<i64>,
        kernel_size: Vec<i64>,
        highways: i64,
        dropout: f64
    ) -> Self {

        let char_level = CharLevelNet::new(vars, char_vocab_size, char_embedding_dim, in_channels, out_channels, kernel_size, highways, in_dim);
        let forward_lm = UniLM::new(vars, n_lstm_layers, in_dim, hidden_dim, dropout);
        let backward_lm = UniLM::new(vars, n_lstm_layers, in_dim, hidden_dim, dropout);
        let to_vocab = nn::linear(vars, in_dim, token_vocab_size, Default::default());

        Self {
            forward_lm: forward_lm,
            backward_lm: backward_lm,
            to_vocab: to_vocab,
            n_lstm_layers: n_lstm_layers,
            char_level: char_level
        }


    }
}

impl ModuleT for ELMo {

    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        
        let dims = xs.internal_shape_as_tensor();
        let dims = Vec::<i64>::from(dims);
        println!("{:?}", dims);

        
        // xs is of shape (batch_size, sequence_length, token_length), batch_size = 1
        // move through char enconding => (batch_size, sequence_length, out_linear)
        let xs_embedded = &self.char_level.forward_t(xs, train);

        // xs_embedded should be (batch_size, sequence_length, out_linear)
        let xs_embedded_flip = Tensor::flip(&xs_embedded.to_owned().shallow_clone(), &[1]);

        // both should be (n_lstm_layers, batch_size, sequence_length, out_linear)
        let forward_lm_outs = self.forward_lm.forward_t(xs_embedded, train);
        let backward_lm_outs = self.backward_lm.forward_t(&xs_embedded_flip, train);

        // the elmo representation is a combination of all the outputs (2L) + xs
        // for the simple case, I take the sum of xs and the two last outputs
        let forward_last = forward_lm_outs.slice(0, self.n_lstm_layers-1, self.n_lstm_layers, 1);
        let backward_last = backward_lm_outs.slice(0, self.n_lstm_layers-1, self.n_lstm_layers, 1);
        
        // all the representations most transfer to vocabulary size
        let xs_embedded_out = xs_embedded.apply(&self.to_vocab);
        let forward_out = forward_last.apply(&self.to_vocab);
        let backward_out = backward_last.apply(&self.to_vocab);

        let out = xs_embedded_out + forward_out + backward_out;
        out // (batch_size, sequence_length, token_vocab_size)

    }
}
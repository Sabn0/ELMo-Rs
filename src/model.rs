
use std::iter::zip;

use tch::{nn, Tensor};
use tch::nn::ModuleT;


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
        println!("weight: {:?}", conv.ws.internal_shape_as_tensor());
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
struct Highway {}

impl ModuleT for Highway {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.tanh()
    }
}

#[derive(Debug)]
pub struct CharLevelNet {
    embedding: nn::Embedding,
    conv_blocks: Vec<CnnBlock>
}

impl CharLevelNet {
    pub fn new(vars: &nn::Path,
         vocab_size: i64, 
         embedding_dim: i64, 
         in_channels: i64, 
         out_channels: Vec<i64>, 
         kernel_size: Vec<i64>) -> Self {

            // out_channels = number of filters
            // kernel_size = matching kernel width

        let embedding = nn::embedding(vars, vocab_size,  embedding_dim, Default::default());
        let mut conv_blocks = Vec::new();
        for (out_channel, kernel_size) in zip(out_channels, kernel_size) {
            let conv_block = CnnBlock::new(vars, in_channels, out_channel, kernel_size);
            conv_blocks.push(conv_block);
        }

        
        Self {
            embedding: embedding,
            conv_blocks: conv_blocks
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

            // each output in outputs is of shape k * (batch_size, n_filters) => (batch_size, total_filters)
            let token_outputs = Tensor::concat(&token_outputs, 1);
            outputs.push(token_outputs);
        }

        // (sequence_length, batch_size, total_filters) => (batch_size, sequence_length, total_filters)
        let outputs = Tensor::concat(&outputs, 0).reshape(&[*batch_size, *seq_length, -1]);
        outputs

    }
}

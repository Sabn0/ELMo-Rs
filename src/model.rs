
use tch::{nn, Tensor};
use tch::nn::ModuleT;


// If a word is of length k charachters, the convolution is What's described as
// a narrow convolution between a charachter within a word, C_k \in (d, l),  
// and a kernel H \in (d, w), where d is char embedding, l is the length of the word,
// and w is the size of the kernal. The multiplication is done as a dot product.
#[derive(Debug)]
pub struct CnnBlock {
    conv: nn::Conv2D
}

impl CnnBlock {
    fn new(vars: &nn::Path, in_channels: i64, out_channels: i64, kernel_size: i64) -> Self {

        let conv = nn::conv2d(vars, in_channels, out_channels, kernel_size, Default::default());

        Self {
            conv: conv
        }

    }
}

impl ModuleT for CnnBlock {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {

        let _dims = xs.internal_shape_as_tensor();

        let conv_out = xs.apply(&self.conv);
        let act_out = conv_out.tanh();
        let pool_out = act_out.max_pool2d_default(5);
        pool_out
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
    conv_block: CnnBlock
}

impl CharLevelNet {
    pub fn new(vars: &nn::Path, vocab_size: i64, embedding_dim: i64, in_channels: i64, out_channels: i64, kernel_size: i64) -> Self {

        let embedding = nn::embedding(vars, vocab_size,  embedding_dim, Default::default());
        let conv_block = CnnBlock::new(vars, in_channels, out_channels, kernel_size);
        
        Self {
            embedding: embedding,
            conv_block: conv_block
        }

    }
    

}

impl ModuleT for CharLevelNet {
    
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {

        // xs is of shape (batch_size, token_length)
        // xs expnsion: (batch_size, token_length) => (batch_size, token_length, embedding_dim)
        println!("{:?}", xs);
        println!("{:?}", xs.dim());
        println!("{:?}", xs.internal_shape_as_tensor());
        let x = xs.apply(&self.embedding);
        println!("{:?}", x.internal_shape_as_tensor());
        let out = self.conv_block.forward_t(&x, train);
        out
        
    }
}



pub mod data_loading {

    use std::collections::HashMap;
    use std::error::Error;
    use tch::Device;
    use tch::IndexOp;
    use tch::Kind;
    use tch::Tensor;
    use crate::config::JsonELMo;

    // a loader similar to Iter2 of tch, but it knows to receive a vector of tensors and not a tensor of tensors
    pub struct Loader {
        xs: Vec<Tensor>,
        ys: Vec<Tensor>,
        device: Device,
        pub batch_size: i64,
        pub seq_length: i64
    }

    impl Loader {
        pub fn new(xs: Vec<Tensor>, ys: Vec<Tensor>, device: Device, batch_size: i64, seq_length: i64) -> Self {
            assert_eq!(xs.len(), ys.len());

            // each element in xs is of shape (sentence_length, max_token_length)
            // each element in ys is of shape (sentence_length)
            
            // Since every sentence has different length, Loader is then followed by StreamLoader, that generates
            // batches of equal length strings (batch_size, seq_length, ...)

            Self {
                xs: xs,
                ys: ys,
                device: device,
                batch_size: batch_size,
                seq_length: seq_length
            }
        }

        pub(in crate) fn shuffle(&mut self) -> &mut Loader {

            // shuffles xs and vs tensors together with random permutation and sends self back

            let n_samples = self.xs.len();
            let permutation = Vec::<i64>::try_from(Tensor::randperm(n_samples as i64, (Kind::Int64, self.device))).unwrap();

            self.xs = (&permutation).into_iter().map(|i| self.xs.get(*i as usize).unwrap().shallow_clone()).collect::<Vec<Tensor>>();
            self.ys = (&permutation).into_iter().map(|i| self.ys.get(*i as usize).unwrap().shallow_clone()).collect::<Vec<Tensor>>();
            assert!(self.xs.len() == n_samples);
            self

        }

        pub(in crate) fn to_stream(&mut self) -> StreamLoader {

            // converts vectors of tensors (xs, ys) to initalized StreamLoader that receives pure tensors

            let xs = Tensor::concat(&self.xs, 0); // of shape (N_tokens, max_token_length)
            let ys = Tensor::concat(&self.ys, 0); // of shape (N_tokens)

            let dims_xs = Vec::<i64>::try_from(xs.internal_shape_as_tensor()).unwrap();
            let dims_ys = Vec::<i64>::try_from(ys.internal_shape_as_tensor()).unwrap();
            assert_eq!(dims_xs[0], dims_ys[0]);
            
            StreamLoader { 
                xs: xs.shallow_clone(), 
                ys: ys.shallow_clone(), 
                device: self.device, 
                batch_size: self.batch_size,
                seq_length: self.seq_length,
                max_token_length: dims_xs[1],
                start_index: 0, 
                end_index: dims_xs[0]
            }
        }

    }

    pub(in crate) struct StreamLoader {
        xs: Tensor,
        ys: Tensor,
        device: Device,
        batch_size: i64,
        seq_length: i64,
        max_token_length: i64,
        start_index: i64,
        end_index: i64
    }

    // implementation of Iterator to a StreamLoader.
    // The iterator generates pairs of (batch_size, seq_length, ...) tensor inside a training iteration. 
    impl Iterator for StreamLoader {
        type Item = (Tensor, Tensor);

        fn next(&mut self) -> Option<Self::Item> {
            
            // stop condition -> ends loop over examples
            if self.start_index >= self.end_index {
                return None
            }

            // slice is the size of examples loaded each next()
            let slice = self.batch_size * self.seq_length;
            let mut end_batch = self.start_index + slice;

            // that handles last smaller batch
            if end_batch > self.end_index {

                // in this case, skipping the less-than seq_length words

                end_batch = self.end_index; 
                let mut xs_batch = self.xs.i(self.start_index..end_batch).to_kind(Kind::Int64).to_device(self.device);
                let mut ys_batch = self.ys.i(self.start_index..end_batch).to_kind(Kind::Int64).to_device(self.device);

                // xs shape is (N, max_token_length). We want to reshape to roughly have dim1 = seq_length
                let dims = Vec::<i64>::try_from(xs_batch.internal_shape_as_tensor()).unwrap();
                if dims[1] < self.seq_length {

                    return None

                } else {

                    // find the biggest number that devides with seq length up to N
                    for i in (dims[1]-self.seq_length..dims[1]).rev() {
                        if i % self.seq_length == 0 {

                            xs_batch = xs_batch.i(..i).reshape([-1, self.seq_length, self.max_token_length]);
                            ys_batch = ys_batch.i(..i).reshape([-1, self.seq_length]);
                            break;
                        }
                    }
                }

                // promote starting index for following next()
                self.start_index = end_batch;

                Some((xs_batch, ys_batch))

            } else {

                // in this case, get the batch and reshape to (batch_size, seq_length, ... )

                let xs_batch = self.xs.i(self.start_index..end_batch).reshape(&[self.batch_size, self.seq_length, -1]).to_kind(Kind::Int64).to_device(self.device); // (batch_size, seq_length, max_token_length)
                let ys_batch = self.ys.i(self.start_index..end_batch).reshape(&[self.batch_size, self.seq_length]).to_kind(Kind::Int64).to_device(self.device); // (batch_size, seq_length)    

                // promote starting index for foloowing next()
                self.start_index = end_batch;

                Some((xs_batch, ys_batch))
            }

            // xs_batch should be (batch_size, seq_length, max_token_length)
            // ys_batch should be (batch_size, seq_length)
            // last iteration might be smaller


        }
    }

    // similar to the pytorch implementation, trait to get an example by its index
    pub trait DatasetBuilder { 
        type Error;
        fn get_len(&self) -> i64;
        fn get_example(&self, index: usize) -> Result<(Tensor, Tensor), Self::Error>;
    }

    pub struct ELMoText {
        sentences: Vec<String>,
        token2int: HashMap<String, usize>,
        char2int: HashMap<char, usize>,
        max_len_token: usize,
        char_start: char,
        char_end: char,
        char_unk: char,
        str_unk: String
    }

    impl ELMoText {
        pub fn new(sentences: Vec<String>, 
            token2int: HashMap<String, usize>, 
            char2int: HashMap<char, usize>, 
            params: &JsonELMo) -> Self {
            
            Self {
                sentences: sentences,
                token2int: token2int,
                char2int: char2int,
                max_len_token: params.max_len_token as usize,
                char_start: params.char_start,
                char_end: params.char_end,
                char_unk: params.char_unk,
                str_unk: params.str_unk.to_string()
            }
        }
    }

    impl DatasetBuilder for ELMoText {

        type Error = Box<dyn Error>;

        fn get_len(&self) -> i64 { 
            self.sentences.len() as i64
        }

        fn get_example(&self, index: usize) -> Result<(Tensor, Tensor), Self::Error> {
            
            // Tensor for chars: each element in the tensor is a tensor of char encodings.
            // the output is of shape (n, max_len_token), n is the length of the sentence.

            // Tensor for labels: each element in the tensor is a label of a token in the sentence.
            // the output is of shape (n, 1), n is the length of the sentence.
            
            // n will be the same for an example and its matching labels. The example will miss 
            // its last element, labels will miss the first label.

            let mut inputs: Vec<Tensor> = Vec::new();
            let example = self.sentences.get(index).ok_or("example index not found in examples indices")?;            

            let map_chars_to_ints = | token: &Vec<char>| -> Vec<i64> {

                // map a token to a series of char ids
                // replace uknown chars with unk char symbol
                let unk_char_id = self.char2int.get(&self.char_unk).expect("didn't find unk char symbol");
                let mut char_ids = token.into_iter().map(|c| {
                    let char_id = self.char2int.get(c).unwrap_or(unk_char_id);
                    *char_id as i64
                    // replacing unknown chars with unk char symbol, not handling seq bytes
                }).collect::<Vec<i64>>();
                
                // obey to max_len_token with pad or truncate
                // pad done with ' '
                let token_len = char_ids.len();
                let pad = *self.char2int.get(&' ').expect("didn't find pad symbol") as i64;
                if self.max_len_token <= token_len {
                    char_ids.truncate(self.max_len_token);
                } else {
                    for _ in token_len..self.max_len_token {
                        char_ids.push(pad);
                    }
                }
                char_ids

            };

            let tokens = example.clone().split(" ").map(|x| x.trim().to_owned()).collect::<Vec<String>>();
            let unk_id = self.token2int.get(&self.str_unk).expect("didn't find unk token symbol");
            let mut labels = (&tokens).iter().map(|t| {
                let label = self.token2int.get(t).cloned().unwrap_or(*unk_id);
                Tensor::from_slice(&[label as i64])
            } ).collect::<Vec<Tensor>>();

            // move each token from string of chars to int encoding of fixed maximal length
            for token in &tokens {
                let mut token_vec = token.split("").filter(|x| x.len()>0).map(|x| x.chars().nth(0).unwrap()).collect::<Vec<char>>();
                token_vec.insert(0, self.char_start);
                token_vec.push(self.char_end);

                let char_ids = map_chars_to_ints(&token_vec);
                let char_tensor = Tensor::from_slice(&char_ids);
                inputs.push(char_tensor);
            }

            // now, inputs is a vec of tensors, each element is a tensor with a series of ints that represent a token.
            // to keep in mind that we will predict the 1 token from the 0 token, 2 from 1, ... n-1 from n-2.
            // so we don't use the last token as an input, and don't use the first token as a label

            let n = inputs.len();
            let _ = labels.remove(0);
            let _ = inputs.remove(n-1);
            assert_eq!(inputs.len(), labels.len());

            // move to tensors
            let inputs_tensor = Tensor::concat(&inputs, 0).reshape(&[-1, self.max_len_token as i64]);
            let labels_tensor = Tensor::concat(&labels, 0).reshape(&[-1]);
            let input_length = Vec::<i64>::try_from(inputs_tensor.internal_shape_as_tensor()).unwrap()[0];
            let labels_length = Vec::<i64>::try_from(labels_tensor.internal_shape_as_tensor()).unwrap()[0];
            assert_eq!(input_length, labels_length);
            
            // inputs_tensor is of shape (sentence_length-1, max_token_length)
            // labels_tnesor is of shape (sentence_length-1)
            let output = (inputs_tensor, labels_tensor);
            Ok(output)

        }
    }


    // An implementation to get a random permutation that is split to train, dev and test sets indices
    // given N number of samples in the corpus
    pub struct Splitter;
    impl Splitter {

        pub fn new() -> Self {
            Self {}
        }

        fn get_split_train_dev_test_ratio(&self) -> [f64; 3] {
            let split_ratio = [0.8, 0.1, 0.1]; // train, dev and test
            assert!(split_ratio.iter().sum::<f64>() == 1.0, "ratios must sum to 1");
            split_ratio
        }

        fn get_split_train_dev_test_sizes(&self, n_samples: i64) -> Vec<i64> {

            let split_ratio = self.get_split_train_dev_test_ratio();
            let mut split_points = vec![
                (split_ratio[0] * n_samples as f64) as i64,
                (split_ratio[1] * n_samples as f64) as i64,
            ];
            split_points.push(n_samples - split_points.iter().sum::<i64>());

            assert!(split_points.iter().sum::<i64>() == n_samples, "number of samples must be equal to sum of splits");
            split_points

        }

        pub fn get_split_train_dev_test_indices(&self, n_samples: i64) -> Vec<Tensor> {
            
            assert!(n_samples > 0, "number of samples for training most be positive");
            
            let split_points: Vec<i64> = self.get_split_train_dev_test_sizes(n_samples);
            let indices: Tensor = Tensor::randperm(n_samples, (Kind::Int64, Device::Cpu));
            let split_indices: Vec<Tensor> = indices.split_with_sizes(&split_points, 0);
            split_indices
        }

    }

}

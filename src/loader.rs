
/*

consider having a custom error for this module
 */


mod data_loading {

    use std::collections::HashMap;
    use std::error::Error;

    use rayon::str::Split;
    use rayon::str::SplitWhitespace;
    use tch::Device;
    use tch::Kind;
    use tch::data::TextData;
    use tch::data::TextDataIter;
    use tch::Tensor;
    use tch::vision::dataset::Dataset;


    pub(in crate) trait DatasetBuilder {
        type Error;
        fn get_len(&self) -> u64;
        fn get_example(&self, index: usize) -> Result<Vec<Tensor>, Self::Error>;
    }

    struct ELMoText {
        examples: Vec<String>,
        char2int: HashMap<char, u8>,
        max_len_token: Option<usize>,
        char_start: char,
        char_end: char
    }

    impl DatasetBuilder for ELMoText {

        type Error = Box<dyn Error>;

        fn get_len(&self) -> u64 { 
            self.examples.len() as u64
        }

        fn get_example(&self, index: usize) -> Result<Vec<Tensor>, Self::Error> {
            
            let mut tensors: Vec<Tensor> = Vec::new();
            let example = self.examples.get(index).ok_or("example index not found in examples indices")?;            

            let map_chars_to_ints = | token: &Vec<char>| -> Vec<u8> {

                // map a token to a series of char ids
                // replace uknown chars with sequence of bytes
                let mut char_ids = token.into_iter().map(|c| {
                    let char_id = self.char2int.get(c).ok_or({
                        let mut char_buf: [u8; 2] = [0; 2]; 
                        c.encode_utf8(&mut char_buf);
                    }).unwrap().to_be();
                    char_id
                }).collect::<Vec<u8>>();
                
                // obey to max_len_token with pad or truncate
                // pad done with ' '
                let token_len = char_ids.len();
                match self.max_len_token {
                    None => {},
                    Some(max_len_token) => {
                        if max_len_token <= token_len {
                            char_ids.truncate(max_len_token);
                        } else {
                            for _ in token_len..max_len_token {
                                char_ids.push(b' ');
                            }
                        }
                    }
                };
                char_ids

            };

            let tokens = example.clone().split(" ").map(|x| x.to_owned()).collect::<Vec<String>>();
            // move each token from string of chars to int encoding of fixed maximal length
            // wrap token with SOT and EOT (SOT is $, EOT is ^), pad with with spaces or truncate.
            for token in &tokens {
                
                let mut token_vec = token.split("").map(|x| x.chars().nth(0).unwrap()).collect::<Vec<char>>();
                token_vec.insert(0, self.char_start);
                token_vec.push(self.char_end);

                let char_ids = map_chars_to_ints(&token_vec);
                let char_tensor = Tensor::of_slice(&char_ids);
                tensors.push(char_tensor);
            }

            Ok(tensors)

        }
    }


    struct loader;
    impl loader {

        fn new() -> Self {
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

            assert!(split_points.iter().sum::<i64>() == n_samples, "number of samples must equate to sum of splits");
            split_points

        }

        fn get_split_train_dev_test_indices(&self, n_samples: i64) -> Vec<Tensor> {
            
            assert!(n_samples > 0, "number of samples for training most be positive");
            
            let split_points: Vec<i64> = self.get_split_train_dev_test_sizes(n_samples);
            let indices: Tensor = Tensor::randperm(n_samples, (Kind::Int8, Device::Cpu));
            let split_indices = indices.split_with_sizes(&split_points, 0);
            split_indices
        }


        fn load() {



        }
    }

}

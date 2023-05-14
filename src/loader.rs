
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
        token2int: HashMap<String, u8>,
        char2int: Option<HashMap<String, u8>>,
        max_len_sentence: Option<usize>,
        max_len_token: Option<usize>
    }

    impl DatasetBuilder for ELMoText {

        type Error = Box<dyn Error>;

        fn get_len(&self) -> u64 { 
            self.examples.len() as u64
        }

        fn get_example(&self, index: usize) -> Result<Vec<Tensor>, Self::Error> {
            
            let mut tensors: Vec<Tensor> = Vec::new();
            let example = self.examples.get(index).ok_or("example index not found in examples indices")?;            

            let map_str_to_int = |
            strs: &Vec<String>, str2int: &HashMap<String, u8>, str_type: &str, max_len: Option<usize>
            | -> Vec<u8> {

                let sos = self.token2int.get("<SOS_{str_type}").ok_or("SOS not found").unwrap();
                let eos = self.token2int.get("<EOS_{str_type}>").ok_or("EOS not found").unwrap();
                let unk = self.token2int.get("<UNK_{str_type}>").ok_or("UNK not found").unwrap();
                let pad = self.token2int.get("<PAD_{str_type}>").ok_or("UNK not found").unwrap();

                // map tokens strings to token ids
                // replace uknown tokens with unk id.  
                let mut str_ids = strs.into_iter().map(|str_| {
                    let str_id = str2int.get(str_).ok_or(unk).unwrap().to_be();
                    str_id
                }).collect::<Vec<u8>>();
                // wrap with SOS and EOS.
                str_ids.insert(0, sos.to_be());
                str_ids.push(eos.to_be());
                // obey to max_len_sentence with  pad or truncate
                let n_str_ids = str_ids.len();
                match max_len {
                    None => {},
                    Some(max_len) => {
                        if max_len <= n_str_ids {
                            str_ids.truncate(max_len);
                        } else {
                            for _ in n_str_ids..max_len {
                                str_ids.push(pad.to_be());
                            }
                        }
                    }
                };
                str_ids

            };

            let tokens = example.clone().split(" ").map(|x| x.to_owned()).collect::<Vec<String>>();

            let token_ids = map_str_to_int(&tokens, &self.token2int, "TOKEN", self.max_len_sentence);
            let token_tensor = Tensor::of_slice(&token_ids);
            tensors.push(token_tensor);

            if self.char2int.is_some() {
                for token in &tokens {
                    let token_vec = token.split("").map(|x| x.to_string()).collect::<Vec<String>>();
                    let char_ids = map_str_to_int(&token_vec, self.char2int.as_ref().unwrap(), "CHAR", self.max_len_token);
                    let char_tensor = Tensor::of_slice(&char_ids);
                    tensors.push(char_tensor);
                }
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

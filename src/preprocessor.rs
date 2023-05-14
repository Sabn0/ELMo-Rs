
// This module handles data preprocessing, matching to the data in the Chelba et al., 2014 benchmark
// Given a vector of strings (the input corpus), the preprocessing stage includes the removal of identical
// sentences, adding SOS and EOS, mapping words under 3 occurrences to UNK, (tokenization?
// randomization?). then, tokenizing done by split and vocabulary is made by tokens. The new sentences vector
// is the output of this preprocessor. Order randomization will be done in the training stage before epochs.

mod preprocessor {

    use std::collections::HashMap;

    use counter::Counter;
    use itertools::Itertools;
    pub(in crate) struct Preprocessor {}

    impl Preprocessor {

        fn new() -> Self { Self { } }

        // remove duplicated sentences, mutate the corpus in self
        fn unique(&self, sentences: &mut Vec<String>) {
            let reduced_sentences = sentences.clone().into_iter().unique().collect::<Vec<String>>();
            *sentences = reduced_sentences;
        }

        // Counter is treated as a hash map.
        // rust strings are vectors, thus there should not be a problem to join X sentences to one variable of 
        // type String, if the first one fits in memory. In other words, if we were able to load X sentences
        // into a Vec<String>, those should fit in one chunk. 
        fn count_tokens(&self, sentences: &Vec<String>) -> Counter<String, i8> {
            let token_chunk = sentences.join(" ");
            let token2count: Counter<String, i8> = token_chunk.split_whitespace().map(|x| x.to_string()).collect();
            token2count
        }

        // uses the counter to get a vector of unique chars
        fn count_chars(&self, vocab: &Vec<String>) -> Counter<char, i8> {
            let char_chunk = vocab.join("");
            let char2count: Counter<char, i8> = char_chunk.chars().collect();
            char2count
        }

        fn preprocess(&mut self, sentences: &mut Vec<String>, vocab_size: usize) -> (HashMap<String, u8>, HashMap<String, u8>){

            // strip duplicated sentences
            self.unique(sentences);

            // create vocabulary of vocab_size most common tokens
            let token2count = self.count_tokens(sentences.as_ref());
            let mut vocab: Vec<String> = token2count.most_common_ordered()
            .into_iter().map(|(s, _)| s).collect::<Vec<String>>();
            vocab.truncate(vocab_size);

            // add SOS, EOS and UNK to vocab
            vocab.append(&mut vec!["<SOS>", "<EOS>", "<UNK>"].into_iter().map(|x| x.to_string()).collect_vec());
            let token2int: HashMap<String, u8> = vocab.iter().enumerate().map(|(i, t)| (t.to_string(), i as u8)).collect();

            // crate vocabulary of chars
            let char2count = self.count_chars(&vocab);
            let chars = char2count.into_iter().map(|(c, _)| c).collect::<Vec<char>>();
            let char2int: HashMap<String, u8> = chars.iter().enumerate().map(|(i, c)| (c.to_string(), i as u8)).collect();

            // replace words not in voabulary or under 3 occurrences with <unk>, pad with SOS + EOS
            // will happen in the loader stages, with random shuffling for each iteration (pytorch style)
            (token2int, char2int)
        }

    }

}

 
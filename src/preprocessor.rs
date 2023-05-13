
// This module handles data preprocessing, matching to the data in the Chelba et al., 2014 benchmark
// Given a vector of strings (the input corpus), the preprocessing stage includes the removal of identical
// sentences, adding SOS and EOS, mapping words under 3 occurrences to UNK, (tokenization?
// randomization?). then, tokenizing done by split and vocabulary is made by tokens. The new sentences vector
// is the output of this preprocessor. Order randomization will be done in the training stage before epochs.

mod preprocessor {

    use counter::Counter;
    use itertools::Itertools;
    pub(in crate) struct preprocessor {}

    impl preprocessor {

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
        fn count(&self, sentences: &Vec<String>) -> Counter<String, i8> {
            let token_chunk = sentences.join(" ");
            let token2count: Counter<String, i8> = token_chunk.split_whitespace().map(|x| x.to_string()).collect();
            token2count
        }

        fn preprocess(&mut self, sentences: &mut Vec<String>, vocab_size: usize) {

            // strip duplicated sentences
            self.unique(sentences);

            // create vocabulary of vocab_size most common tokens
            let token2count = self.count(sentences.as_ref());
            let mut vocab: Vec<(String, i8)> = token2count.most_common_ordered();
            vocab.truncate(vocab_size);

            // replace words not in voabulary or under 3 occurrences with <unk>, pad with SOS + EOS
            // will happen in the loader stages, with random shuffling for each iteration (pytorch style)
        }

    }

}

 
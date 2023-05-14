
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

        // uses the counter to get a vector of unique chars
        fn count_chars(&self, vocab: &Vec<String>, char_start: char, char_end: char) -> Counter<char, i8> {
            let char_chunk = vocab.join("");
            let mut char2count: Counter<char, i8> = char_chunk.chars().collect();
            let n = char2count.len();
            char2count.extend([(char_start, n as i8), (char_end, 1 + n as i8)]);
            char2count
        }

        fn preprocess(&mut self, sentences: &mut Vec<String>, vocab_size: usize) -> HashMap<char, u8>{

            // strip duplicated sentences
            self.unique(sentences);

            // pad sentences with SOS + EOS
            let char_start = '$';
            let char_end = '^';
            sentences.iter_mut().map(|s| { // filtering future EOT and SOT chars
                *s = s.trim_matches(' ').to_string(); // remove leading and trailing spaces
                *s = s.chars().filter(|x| x != &char_start && x != &char_end).collect::<String>();
                *s = "SOS ".to_string() + s;
                *s += " EOS";
            });

            // crate vocabulary of chars
            let char2count = self.count_chars(&sentences, char_start, char_end);
            let chars = char2count.into_iter().map(|(c, _)| c).collect::<Vec<char>>();
            let char2int: HashMap<char, u8> = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_owned(), i as u8))
            .collect();

            // char2int has all the chars in the corpus + start + end chars, that has been filtered from the sentences.
            // sentences don't have duplications, padded with SOS and EOS tokens.
            // there is no UNK for chars, an unknown char will be replaced with it's uft8 encoding.

            char2int
        }

    }

}

 

// This module handles data preprocessing, matching to the data in the Chelba et al., 2014 benchmark
// Given a vector of strings (the input corpus), the preprocessing stage includes the removal of identical
// sentences, adding SOS and EOS, mapping words under 3 occurrences to UNK, (tokenization?
// randomization?). then, tokenizing done by split and vocabulary is made by tokens. The new sentences vector
// is the output of this preprocessor. Order randomization will be done in the training stage before epochs.

mod preprocessor {

    use std::collections::HashMap;

    use counter::Counter;
    use itertools::Itertools;

    pub trait CollectT {
        type Item;
        fn collect_gen(elements: Vec<Self::Item>) -> HashMap<Self::Item, u8>;
    }
    impl CollectT for char {
        type Item = Self;
        fn collect_gen(elements: Vec<Self::Item>) -> HashMap<Self::Item, u8> {
            let x: HashMap<char, u8> = elements.
            into_iter()
            .enumerate()
            .map(|(i, e)| (e, i as u8))
            .collect();
            x
        }
    }
    impl CollectT for String {
        type Item = Self;
        fn collect_gen(elements: Vec<Self::Item>) -> HashMap<Self::Item, u8> {
            let x: HashMap<String, u8> = elements.
            into_iter()
            .enumerate()
            .map(|(i, e)| (e, i as u8))
            .collect();
            x
        }
    }

    pub(in crate) struct Preprocessor {}

    impl Preprocessor {

        fn new() -> Self { Self { } }

        // remove duplicated sentences, mutate the corpus in self
        fn unique(&self, sentences: &mut Vec<String>) {
            let reduced_sentences = sentences.clone().into_iter().unique().collect::<Vec<String>>();
            *sentences = reduced_sentences;
        }

        // uses the counter to get a vocabulary of words
        fn count_tokens(&self, sentences: &Vec<String>, vocab_size: usize, min_count: usize) -> Vec<String> {
            let chunk = sentences.join(" ");
            let token2count: Counter<String, i8> = chunk.split_whitespace().map(|x| x.to_string()).collect();
            let tokens = token2count.k_most_common_ordered(vocab_size);
            let tokens = tokens
            .into_iter()
            .filter(|(_, c)| *c as usize >= min_count)
            .map(|(t, _)| t)
            .collect::<Vec<String>>();
            tokens
        }

        // uses the counter to get a vector of unique chars
        fn count_chars(&self, vocab: &Vec<String>, char_start: char, char_end: char) -> Vec<char> {
            let char_chunk = vocab.join("");
            let mut char2count: Counter<char, i8> = char_chunk.chars().collect();
            let n = char2count.len();
            char2count.extend([(char_start, n as i8), (char_end, 1 + n as i8)]);
            let chars = char2count.into_iter().map(|(c, _)| c).collect::<Vec<char>>();
            chars
        }


        fn preprocess(&mut self, sentences: &mut Vec<String>, vocab_size: usize, min_count: usize) -> (HashMap<String, u8>, HashMap<char, u8>) {

            // strip duplicated sentences
            self.unique(sentences);

            // pad sentences with SOS + EOS
            let char_start = '$';
            let char_end = '^';
            sentences.iter_mut().for_each(|s| { // filtering future EOT and SOT chars
                *s = s.trim_matches(' ').to_string(); // remove leading and trailing spaces
                *s = s.chars().filter(|x| x != &char_start && x != &char_end).collect::<String>();
                *s = "SOS ".to_string() + s;
                *s += " EOS";
            });

            // create vocabulary of words
            let tokens = self.count_tokens(&sentences, vocab_size, min_count);
            let token2int: HashMap<String, u8> = <String as CollectT>::collect_gen(tokens);

            // create vocabulary of chars
            let chars = self.count_chars(&sentences, char_start, char_end);
            let char2int: HashMap<char, u8> = <char as CollectT>::collect_gen(chars);

            // char2int has all the chars in the corpus + start + end chars, that has been filtered from the sentences.
            // sentences don't have duplications, padded with SOS and EOS tokens.
            // there is no UNK for chars, an unknown char will be replaced with it's uft8 encoding.

            (token2int, char2int)
        }

    }

}

 
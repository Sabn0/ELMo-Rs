

pub mod do_preprocess {

    use std::collections::HashMap;
    use counter::Counter;
    use itertools::Itertools;
    use crate::config::JsonELMo;

    // collect a Hashmap into a vector of tuples in reversed key order
    pub(in crate) trait CollectT {
        type Item;
        fn collect_gen(elements: Vec<Self::Item>) -> HashMap<Self::Item, usize>;
    }
    impl CollectT for char {
        type Item = Self;
        fn collect_gen(elements: Vec<Self::Item>) -> HashMap<Self::Item, usize> {
            let x: HashMap<char, usize> = elements.
            into_iter()
            .enumerate()
            .map(|(i, e)| (e, i as usize))
            .collect();
            x
        }
    }
    impl CollectT for String {
        type Item = Self;
        fn collect_gen(elements: Vec<Self::Item>) -> HashMap<Self::Item, usize> {
            let x: HashMap<String, usize> = elements.
            into_iter()
            .enumerate()
            .map(|(i, e)| (e, i as usize))
            .collect();
            x
        }
    }

    pub struct Preprocessor;
    impl Preprocessor {

        pub fn new() -> Self { Self {} }

        // remove duplicated sentences, mutate the sentences in self
        fn unique(&self, sentences: &mut Vec<String>) {
            let reduced_sentences = sentences.clone().into_iter().unique().collect::<Vec<String>>();
            *sentences = reduced_sentences;
        }

        // uses the counter to get a vector of unique words
        fn count_tokens(&self, sentences: &Vec<String>, token_vocab_size: &mut i64, min_count: i64, str_unk: &str) -> Vec<String> {

            println!("counting from {} sentences", sentences.len());
            let chunk = sentences.join(" ");
            let token2count = chunk.split_whitespace().map(|x| x.to_string()).collect::<Counter<_>>();
            let mut tokens = token2count.k_most_common_ordered(*token_vocab_size as usize);
            tokens.push((String::from(str_unk), tokens.len()));
            let tokens = tokens
            .into_iter()
            .filter(|(_, c)| *c as i64 >= min_count)
            .map(|(t, _)| t)
            .collect::<Vec<String>>();
            *token_vocab_size = tokens.len() as i64;
            println!("working on token vocab : {}", *token_vocab_size);
            tokens
        }

        // uses the counter to get a vector of unique chars
        fn count_chars(&self, vocab: &Vec<String>, char_vocab_size: &mut i64, char_start: char, char_end: char, char_unk: char) -> Vec<char> {
            let char_chunk = vocab.join("");
            let mut char2count = char_chunk.chars().collect::<Counter<_>>().k_most_common_ordered(*char_vocab_size as usize);
            let n = char2count.len();
            char2count.extend([(char_start, n), (char_end, 1 + n), (char_unk, 2 + n)]);
            let chars = char2count.into_iter().map(|(c, _)| c).collect::<Vec<char>>();
            *char_vocab_size = chars.len() as i64;
            println!("working on char vocab : {}", *char_vocab_size);
            chars
        }


        pub fn preprocess(&mut self, sentences: &mut Vec<String>, params: &mut JsonELMo) -> (HashMap<String, usize>, HashMap<char, usize>) {

            // extract elmo parameters
            let token_vocab_size = &mut params.token_vocab_size;
            let char_vocab_size = &mut params.char_vocab_size;
            let min_count = params.min_count;
            let char_start = params.char_start;
            let char_end = params.char_end;
            let char_unk = params.char_unk;
            let str_unk = &params.str_unk;

            // strip duplicated sentences
            self.unique(sentences);

            // some string work on sentences 
            sentences.iter_mut().for_each(|s| { 
                *s = s.trim_matches(' ').to_string(); // remove leading and trailing spaces
                *s = s.chars().filter(|x| x != &char_start && x != &char_end && x != &char_unk).collect::<String>(); // filtering future EOT and SOT chars
                *s = "SOS ".to_string() + s;             // pad sentences with SOS + EOS symbols 
                *s += " EOS";
            });

            // create vocabulary of words
            let tokens = self.count_tokens(&sentences, token_vocab_size, min_count, str_unk);
            let token2int: HashMap<String, usize> = <String as CollectT>::collect_gen(tokens);

            // create vocabulary of chars
            let chars = self.count_chars(&sentences, char_vocab_size, char_start, char_end, char_unk);
            let char2int: HashMap<char, usize> = <char as CollectT>::collect_gen(chars);

            // token2int is bound with vocab_size tokens, minimum occurrences of min count. 
            // It countains and SOS, EOS and UNK tokens. 
            // char2int has all the chars in the corpus + start + end chars + unk char, that has been filtered from the sentences.
            
            (token2int, char2int)
        }

    }

}

 
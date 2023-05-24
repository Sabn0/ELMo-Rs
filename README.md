# GloVe Trainer in Rust

![example workflow](https://github.com/Sabn0/ELMo-Rs/actions/workflows/rust.yml/badge.svg)
![rust version](https://img.shields.io/badge/rust-1.69.0-blue)

An rust implementation of the architecture of the **ELMo** model based on the details in the [ELMo's paper](https://arxiv.org/pdf/1802.05365.pdf). I implemented it with a main focus on experimenting with the [tch](https://crates.io/crates/tch) crate. The main.rs binary works in a 
train-dev-test setting on toy data, but I made no testing over large amounts of data.

 ## Details
If wished, you run a training process after cloning the repo using :
 ```
 cargo build --release
./target/release/main args.json
 ```
 The *main.rs* binary expects a single argument, a json file. The json should specify (at the minimum) two parameters: (1) a txt file with corpus of sentences for training. (2) a location for output file that will save a trained model. As follows:
 ```javascript
 {
    "corpus_file": "Input/corpus.txt",
    "output_dir": "Output/model"
 }
 ```
The program will run with the default parameters, that can also be changed usng the json file. Input corpus will be split to train, dev and test sets.

I didn't test the code on any real data, my focus was on the model architecture and the tch crate usage. In particular the training process
lacks some details. For example there is no support for multi-threading in the training process.


## References
* This is a rust implementation of the ELMo architecture described in the [paper](https://aclanthology.org/N18-1202/), by <ins>Matthew E. Peters at al., 2018</ins>.


## Software
Rust version 1.69.0, see Cargo.toml file for the packages I used.


## License
Under [MIT license](...) and [Apache License, Version 2.0](...).
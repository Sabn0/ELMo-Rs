
// imports

use serde_json::Value;
use tch::Device;
use std::{fs::{self}, error::Error, fmt::Display};


#[derive(Clone, Debug)]
pub struct JsonELMo {
    pub corpus_file: String,
    pub output_dir: String,
    pub token_vocab_size: i64,
    pub char_vocab_size: i64,
    pub min_count: i64,
    pub max_len_token: i64,
    pub char_start: char,
    pub char_end: char,
    pub char_unk: char,
    pub str_unk: String,
    pub batch_size: i64,
    pub char_embedding_dim: i64,
    pub in_channels: i64,
    pub out_channels: Vec<i64>,
    pub kernel_size: Vec<i64>,
    pub highways: i64,
    pub in_dim: i64,
    pub hidden_dim: i64,
    pub n_lstm_layers: i64,
    pub dropout: f64,
    pub device: Device,
    pub max_iter: i64,
    pub learning_rate: f64
}

impl Display for JsonELMo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "using hyper-params:
        token_vocab_size: {}
        char_vocab_size: {}
        min_count: {}
        max_len_token: {}
        char_embedding_dim: {},
        in_channels: {},
        out_channels: {:?},
        kernel_size: {:?},
        highways: {},
        in_dim: {},
        hidden_dim: {},
        n_lstm_layers: {},
        device: {:?},
        max_iter: {},
        learning_rate: {}",
        self.token_vocab_size,
        self.char_vocab_size, 
        self.min_count, 
        self.max_len_token, 
        self.char_embedding_dim, 
        self.in_channels, 
        self.out_channels, 
        self.kernel_size, 
        self.highways, 
        self.in_dim, 
        self.hidden_dim, 
        self.n_lstm_layers, 
        self.device, 
        self.max_iter, 
        self.learning_rate)
    }
}

// validation of input arguments
pub struct ConfigElmo {
    params: JsonELMo
}

impl ConfigElmo {

    // program should receive one arguments, path to json
    pub fn new(args: &[String]) -> Result<ConfigElmo, Box<dyn Error>> { 

        if args.len() != 2 {
            return Err(format!("input should be a path to json file only").into());
        }

        let json = ConfigElmo::read_json(&args[1]);
        let params = ConfigElmo::validate(json)?;

        Ok (
            Self {
                params: params
            }
        )
    }

    pub fn get_params(&self) -> JsonELMo {
        return self.params.clone()
    }

}

pub trait Conigure {
    type Item;
    fn read_json(json_path: &String) -> Value;
    fn defaults(corpus_file: String, output_dir: String) -> Self::Item;
    fn validate(json: Value) -> Result<Self::Item, Box<dyn Error>>;
}

impl Conigure for ConfigElmo {

    type Item = JsonELMo;

    fn read_json(json_path: &String) -> Value {
        let f = fs::File::open(json_path).expect("cannot open json file");
        let json: Value = serde_json::from_reader(f).expect("cannot read json file");
        json
    }

    fn defaults(corpus_file: String, output_dir: String) -> Self::Item {

        Self::Item {
            token_vocab_size: 300_000,
            char_vocab_size: 128,
            min_count: 3,
            max_len_token: 50,
            char_embedding_dim: 15,
            in_channels: 1,
            kernel_size: vec![1, 2], // small vec![1,2,3,4,5,6]
            out_channels: vec![25, 50], // small vec![25,50,75,100,125,150]
            highways: 1,
            in_dim: 200, // small 512
            hidden_dim: 400, // small 4096
            n_lstm_layers: 2,
            dropout: 0.1,
            max_iter: 10,
            learning_rate: 0.001, // maybe different
            device: Device::cuda_if_available(),
            char_start: '$',
            char_end: '^',
            char_unk: '~',
            str_unk: String::from("UNK"),
            batch_size: 1,
            corpus_file: corpus_file,
            output_dir: output_dir,
        }

    }

    // the input json has many fields of hyper parameters, this function mainly performs checks 
    // for data types.
    fn validate(json: Value) -> Result<Self::Item, Box<dyn Error>> {

        let validate_str = |field: &str| {
            json.get(field)
            .expect(format!("{} was not supplied throught json", field).as_str())
            .as_str()
            .expect(format!("cannot cast {} to string", field).as_str())
        };

        let validate_float = |field: &str| -> Result<f64, Box<dyn Error>> {
            json.get(field).ok_or("field not given")?.as_f64().ok_or("not float".into())
        };

        let validate_positive_int = |field: &str| -> Result<i64, Box<dyn Error>> {
            let val = json.get(field).ok_or("field not given")?.as_u64().ok_or::<String>("not int".into())?;
            if val ==  0 { return Err(format!("not positive int").into()) }
            Ok(val as i64)
        };

        let validate_vec = |field: &str| -> Result<Vec<i64>, Box<dyn Error>> {
            let arr = json.get(field).ok_or("field not given")?.as_array().ok_or::<String>("not vec".into())?;
            let mut values = Vec::new();
            for val in arr {
                let val = val.get(field).ok_or("field not given")?.as_u64().ok_or::<String>("not int".into())?;
                if val ==  0 { return Err(format!("not positive int").into()) }
                values.push(val as i64);
            }
            Ok(values)
        };

        // validate input and output in json - most be given
        let corpus_file = validate_str("corpus_file").to_string();
        let output_dir = validate_str("output_dir").to_string();
        let mut params = ConfigElmo::defaults(corpus_file, output_dir);

        // validate optional input parameters
        if let Ok(token_vocab_size) = validate_positive_int("token_vocab_size") {
            params.token_vocab_size = token_vocab_size;
        }
        if let Ok(char_vocab_size) = validate_positive_int("char_vocab_size") {
            params.char_vocab_size = char_vocab_size;
        }
        if let Ok(min_count) = validate_positive_int("min_count") {
            params.min_count = min_count;
        }
        if let Ok(max_len_token) = validate_positive_int("max_len_token") {
            params.max_len_token = max_len_token;
        }
        if let Ok(char_embedding_dim) = validate_positive_int("char_embedding_dim") {
            params.char_embedding_dim = char_embedding_dim;
        }
        if let Ok(in_channels) = validate_positive_int("in_channels") {
            params.in_channels = in_channels;
        }
        if let Ok(highways) = validate_positive_int("highways") {
            params.highways = highways;
        }
        if let Ok(in_dim) = validate_positive_int("in_dim") {
            params.in_dim = in_dim;
        }
        if let Ok(hidden_dim) = validate_positive_int("hidden_dim") {
            params.hidden_dim = hidden_dim;
        }
        if let Ok(n_lstm_layers) = validate_positive_int("n_lstm_layers") {
            params.n_lstm_layers = n_lstm_layers;
        }
        if let Ok(max_iter) = validate_positive_int("max_iter") {
            params.max_iter = max_iter;
        }
        if let Ok(dropout) = validate_float("dropout") {
            params.dropout = dropout;
        }
        if let Ok(learning_rate) = validate_float("learning_rate") {
            params.learning_rate = learning_rate;
        }
        if let Ok(out_channels) = validate_vec("out_channels") {
            params.out_channels = out_channels;
        }
        if let Ok(kernel_size) = validate_vec("kernel_size") {
            params.kernel_size = kernel_size;
        }
        Ok(params)

    }
    
}


pub mod files_handling {

    use std::{io::{Lines, BufReader, self, BufRead}, fs::File, error::Error};

    fn read_file(file_path: &str) -> Result<Lines<BufReader<File>>, Box<dyn Error>> {

        match File::open(file_path) {
            Ok(f) => Ok(io::BufReader::new(f).lines()),
            Err(e) => Err(Box::new(e))
        }
    }

    pub fn load_sentences(file_path: &str) -> Result<Vec<String>, Box<dyn Error>> {

        let lines = read_file(file_path)?;        
        let sentences = lines.into_iter().map(|line| parse_line(line.unwrap())).collect::<Vec<String>>();
        Ok(sentences)
    }

    fn parse_line(line: String) -> String {

        // line is a string of text, it is trimmed for trailing and ending spaces, lower cased.
        // As default add end-of-sequence and start-of-sequence tokens, but can be set to false.
        let line_str = [&line.trim().to_lowercase()].map(|x| x.to_string()).to_vec().join(" ");
        line_str
    }

}
use std::error::Error;


struct Config {}

impl Config {

    fn new(args: &[String]) -> Result<Self, Box<dyn Error>> {

        // program should have one variale, a path to a json file
        assert_eq!(args.len(), 2, "");

        Ok(Self{})

    }

}
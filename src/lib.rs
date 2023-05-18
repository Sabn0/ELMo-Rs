//! A pure rust implementation of the ELMo model, based on the original paper.
//! 
//! 
//! 


mod preprocessor;
mod loader;
mod model;
mod trainer;

pub use loader::data_loading::DatasetBuilder;
pub use loader::data_loading::ELMoText;
pub use loader::data_loading::Loader;
pub use preprocessor::do_preprocess::Preprocessor;
pub use model::ELMo;
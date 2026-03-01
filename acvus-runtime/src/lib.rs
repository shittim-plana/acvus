mod error;
mod storage;
mod value;

pub use value::{PureValue, StorageKey};

pub trait Executable {
    
}

pub struct Module<S> {
    storage: S
}
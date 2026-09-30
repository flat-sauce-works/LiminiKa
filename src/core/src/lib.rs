// Core architecture module declarations
pub mod abi;
pub mod dpsr;
pub mod edbc;
pub mod math;
pub mod srl;
pub mod storage;
pub mod swarm;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

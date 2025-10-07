use std::sync::LazyLock;

pub static LOGICAL_CORES: LazyLock<usize> = LazyLock::new(num_cpus::get);
pub static PHYSICAL_CORES: LazyLock<usize> = LazyLock::new(num_cpus::get_physical);
use std::{
    env, fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    stream::StreamBinding, types::Complex32,
};
use singe_cutensor::{
    mp::{
        context::Context as MpContext,
        plan::{OperationDescriptor, Plan, PlanPreference},
        tensor::{RankPartition, TensorDescriptor, TensorDescriptorConfig, TensorMode},
        types::Algorithm,
    },
    operation::ComputeDescriptor,
    types::{Mode, Operator},
};
use singe_nccl::{communicator::Communicator, types::UniqueId, unique_id};

const DEFAULT_EQUATION: &str = "mk,kn->mn";
const DEFAULT_EXTENT: u64 = 2048;
const DEFAULT_REPETITIONS: usize = 5;
const DEFAULT_DEVICE_WORKSPACE_BUDGET: u64 = 10 * 1024 * 1024 * 1024;
const DEFAULT_HOST_WORKSPACE_BUDGET: u64 = 1024;
const UNIQUE_ID_BYTES: usize = 128;

#[derive(Debug)]
struct Config {
    equation: String,
    dist_a: String,
    dist_b: String,
    dist_c: String,
    extent: u64,
    repetitions: usize,
    rank: i32,
    rank_count: i32,
    local_device: Option<i32>,
    id_file: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            equation: DEFAULT_EQUATION.into(),
            dist_a: String::new(),
            dist_b: String::new(),
            dist_c: String::new(),
            extent: DEFAULT_EXTENT,
            repetitions: DEFAULT_REPETITIONS,
            rank: env_parse("SINGE_CUTENSOR_MP_RANK").unwrap_or(0),
            rank_count: env_parse("SINGE_CUTENSOR_MP_NRANKS").unwrap_or(1),
            local_device: env_parse("SINGE_CUTENSOR_MP_LOCAL_DEVICE"),
            id_file: env::var_os("SINGE_CUTENSOR_MP_ID_FILE")
                .map(PathBuf::from)
                .unwrap_or_else(|| env::temp_dir().join("singe-cutensor-mp-nccl-id.bin")),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    validate_config(&config)?;

    let device_count = Device::count()?;
    let local_device = config
        .local_device
        .unwrap_or_else(|| config.rank.rem_euclid(device_count));

    let cuda_context = CudaContext::retain_primary_for_device(Device::new(local_device))?;
    let stream = cuda_context.create_stream()?;
    let stream_binding = StreamBinding::Borrowed(stream.to_borrowed());

    let communicator = create_communicator(&cuda_context, &config)?;
    let context = MpContext::create(&communicator, local_device, stream_binding)?;

    if config.rank == 0 {
        println!("cuTENSORMp contraction sample");
        println!("rank count: {}", config.rank_count);
        println!("equation: {}", config.equation);
    }

    let [modes_a_chars, modes_b_chars, modes_c_chars] = split_equation(&config.equation)?;
    let modes_a = chars_to_raw_modes(&modes_a_chars);
    let modes_b = chars_to_raw_modes(&modes_b_chars);
    let modes_c = chars_to_raw_modes(&modes_c_chars);
    let cutensor_modes_a = raw_modes_to_cutensor(&modes_a);
    let cutensor_modes_b = raw_modes_to_cutensor(&modes_b);
    let cutensor_modes_c = raw_modes_to_cutensor(&modes_c);

    let extent_a = get_extent(&modes_a, config.extent);
    let extent_b = get_extent(&modes_b, config.extent);
    let extent_c = get_extent(&modes_c, config.extent);

    let distributed_modes_a = build_distributed_modes(&modes_a, &config.dist_a);
    let distributed_modes_b = build_distributed_modes(&modes_b, &config.dist_b);
    let distributed_modes_c = build_distributed_modes(&modes_c, &config.dist_c);

    let ranks_per_mode_a = get_ranks_per_mode(&modes_a, &distributed_modes_a, &extent_a);
    let ranks_per_mode_b = get_ranks_per_mode(&modes_b, &distributed_modes_b, &extent_b);
    let ranks_per_mode_c = get_ranks_per_mode(&modes_c, &distributed_modes_c, &extent_c);

    let local_extent_a = calc_local_extents(&extent_a, &ranks_per_mode_a);
    let local_extent_b = calc_local_extents(&extent_b, &ranks_per_mode_b);
    let local_extent_c = calc_local_extents(&extent_c, &ranks_per_mode_c);

    if config.rank == 0 {
        print_modes("modeA", &modes_a_chars);
        print_modes("modeB", &modes_b_chars);
        print_modes("modeC", &modes_c_chars);
        print_values("extentA", &extent_a);
        print_values("extentB", &extent_b);
        print_values("extentC", &extent_c);
        print_values("nranksPerModeA", &ranks_per_mode_a);
        print_values("nranksPerModeB", &ranks_per_mode_b);
        print_values("nranksPerModeC", &ranks_per_mode_c);
        print_values("localExtentA", &local_extent_a);
        print_values("localExtentB", &local_extent_b);
        print_values("localExtentC", &local_extent_c);
    }
    let tensor_modes_a = extent_a
        .iter()
        .copied()
        .map(TensorMode::contiguous)
        .collect::<Vec<_>>();
    let tensor_modes_b = extent_b
        .iter()
        .copied()
        .map(TensorMode::contiguous)
        .collect::<Vec<_>>();
    let tensor_modes_c = extent_c
        .iter()
        .copied()
        .map(TensorMode::contiguous)
        .collect::<Vec<_>>();

    let descriptor_a = TensorDescriptor::create(
        &context,
        TensorDescriptorConfig {
            modes: &tensor_modes_a,
            partition: RankPartition {
                ranks_per_mode: &ranks_per_mode_a,
                ranks: None,
                rank_count: config.rank_count as u32,
            },
            data_type: DataType::ComplexF32,
        },
    )?;
    let descriptor_b = TensorDescriptor::create(
        &context,
        TensorDescriptorConfig {
            modes: &tensor_modes_b,
            partition: RankPartition {
                ranks_per_mode: &ranks_per_mode_b,
                ranks: None,
                rank_count: config.rank_count as u32,
            },
            data_type: DataType::ComplexF32,
        },
    )?;
    let descriptor_c = TensorDescriptor::create(
        &context,
        TensorDescriptorConfig {
            modes: &tensor_modes_c,
            partition: RankPartition {
                ranks_per_mode: &ranks_per_mode_c,
                ranks: None,
                rank_count: config.rank_count as u32,
            },
            data_type: DataType::ComplexF32,
        },
    )?;

    let operation = OperationDescriptor::create_contraction(
        &context,
        &descriptor_a,
        &cutensor_modes_a,
        Operator::Identity,
        &descriptor_b,
        &cutensor_modes_b,
        Operator::Identity,
        &descriptor_c,
        &cutensor_modes_c,
        Operator::Identity,
        &descriptor_c,
        &cutensor_modes_c,
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create(
        &context,
        Algorithm::Default,
        DEFAULT_DEVICE_WORKSPACE_BUDGET,
        DEFAULT_HOST_WORKSPACE_BUDGET,
    )?;
    if config.rank == 0 {
        println!("cuTENSORMp using algorithm {}", preference.algorithm());
    }

    let plan = Plan::create(&context, &operation, Some(&preference))?;
    let workspace = plan.workspace()?;
    if config.rank == 0 {
        println!(
            "workspace: device={} bytes, host={} bytes",
            workspace.device_size(),
            workspace.host_size()
        );
    }
    let mut workspace_memory = plan.create_workspace_memory()?;

    let mut a = DeviceMemory::<Complex32>::create(product_usize(&local_extent_a)?)?;
    let mut b = DeviceMemory::<Complex32>::create(product_usize(&local_extent_b)?)?;
    let mut c = DeviceMemory::<Complex32>::create(product_usize(&local_extent_c)?)?;

    unsafe {
        a.set_value_async_unchecked((config.rank & 0xff) as u8, &stream)?;
        b.set_value_async_unchecked(
            ((config.rank * config.rank_count + config.rank + 1) & 0xff) as u8,
            &stream,
        )?;
    }
    stream.synchronize()?;

    let alpha = Complex32::new(1.0, 0.0);
    let beta = Complex32::new(0.0, 0.0);
    let mut timings = Vec::new();

    for repetition in 0..config.repetitions {
        unsafe {
            c.set_value_async_unchecked(0, &stream)?;
        }
        stream.synchronize()?;

        let start = Instant::now();
        plan.contract_in_place(&alpha, &a, &b, &beta, &mut c, &mut workspace_memory)?;
        stream.synchronize()?;

        if repetition > 0 {
            timings.push(start.elapsed().as_secs_f64() * 1.0e3);
        }
    }

    if config.rank == 0 && !timings.is_empty() {
        let total_ms = timings.iter().sum::<f64>();
        let min_ms = timings.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "\ncuTENSORMp - average time: {:.3}ms, min time: {:.3}ms",
            total_ms / timings.len() as f64,
            min_ms
        );
        print_performance_metrics(
            &modes_a,
            &modes_b,
            &modes_c,
            &local_extent_a,
            &local_extent_b,
            &local_extent_c,
            min_ms,
        );
    }

    if config.rank == 0 {
        println!("\ndone: everything completed successfully");
    }
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut args = env::args().skip(1).peekable();

    while let Some(arg) = args.next() {
        let (name, value) = if let Some((name, value)) = arg.split_once('=') {
            (name.to_string(), Some(value.to_string()))
        } else {
            (arg, None)
        };

        match name.as_str() {
            "--eq" => config.equation = next_value(value, &mut args, "--eq")?,
            "--distA" => config.dist_a = next_value(value, &mut args, "--distA")?,
            "--distB" => config.dist_b = next_value(value, &mut args, "--distB")?,
            "--distC" => config.dist_c = next_value(value, &mut args, "--distC")?,
            "--extent" => config.extent = next_value(value, &mut args, "--extent")?.parse()?,
            "--reps" => config.repetitions = next_value(value, &mut args, "--reps")?.parse()?,
            "--rank" => config.rank = next_value(value, &mut args, "--rank")?.parse()?,
            "--nranks" => config.rank_count = next_value(value, &mut args, "--nranks")?.parse()?,
            "--local-device" => {
                config.local_device =
                    Some(next_value(value, &mut args, "--local-device")?.parse()?);
            }
            "--id-file" => {
                config.id_file = PathBuf::from(next_value(value, &mut args, "--id-file")?)
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    Ok(config)
}

fn next_value(
    inline: Option<String>,
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(value) = inline {
        return Ok(value);
    }
    args.next()
        .ok_or_else(|| format!("missing value for `{name}`").into())
}

fn print_usage() {
    println!(
        "usage: cargo run -p singe-cutensor --example cutensor_mp_contraction -- [options]\n\
         options:\n\
           --eq <einsum>              contraction equation, default `{DEFAULT_EQUATION}`\n\
           --distA <modes>            distributed modes for A\n\
           --distB <modes>            distributed modes for B\n\
           --distC <modes>            distributed modes for C/D\n\
           --extent <n>               uniform extent per mode, default {DEFAULT_EXTENT}\n\
           --reps <n>                 repetitions, default {DEFAULT_REPETITIONS}\n\
           --rank <rank>              NCCL rank, default env SINGE_CUTENSOR_MP_RANK or 0\n\
           --nranks <n>               NCCL rank count, default env SINGE_CUTENSOR_MP_NRANKS or 1\n\
           --local-device <id>        CUDA device id, default rank modulo device count\n\
           --id-file <path>           NCCL unique-id rendezvous file for multi-process runs"
    );
}

fn validate_config(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    if config.rank_count <= 0 {
        return Err("rank count must be positive".into());
    }
    if config.rank < 0 || config.rank >= config.rank_count {
        return Err("rank must be in [0, rank_count)".into());
    }
    if config.extent == 0 {
        return Err("extent must be positive".into());
    }
    if config.repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    Ok(())
}

fn create_communicator(
    cuda_context: &std::sync::Arc<CudaContext>,
    config: &Config,
) -> Result<Communicator, Box<dyn std::error::Error>> {
    let id = if config.rank_count == 1 {
        unique_id()?
    } else {
        unique_id_from_file(config.rank, &config.id_file)?
    };
    Ok(Communicator::create_rank(
        cuda_context,
        config.rank_count,
        id,
        config.rank,
    )?)
}

fn unique_id_from_file(rank: i32, path: &Path) -> Result<UniqueId, Box<dyn std::error::Error>> {
    if rank == 0 {
        let id = unique_id()?;
        fs::write(path, id.as_bytes())?;
        Ok(id)
    } else {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            match fs::read(path) {
                Ok(bytes) if bytes.len() == UNIQUE_ID_BYTES => {
                    let mut array = [0_u8; UNIQUE_ID_BYTES];
                    array.copy_from_slice(&bytes);
                    return Ok(UniqueId::from_bytes(array));
                }
                Ok(_) => {
                    return Err("invalid nccl unique id file length".into());
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::NotFound
                        && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(50));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

fn split_equation(eq: &str) -> Result<[Vec<char>; 3], Box<dyn std::error::Error>> {
    let compact = eq
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    let Some((lhs, rhs)) = compact.split_once("->") else {
        return Err("equation must contain `->`".into());
    };
    let inputs = lhs.split(',').collect::<Vec<_>>();
    if inputs.len() != 2 || rhs.is_empty() || inputs.iter().any(|input| input.is_empty()) {
        return Err("equation must have form `a,b->c`".into());
    }
    Ok([
        inputs[0].chars().collect(),
        inputs[1].chars().collect(),
        rhs.chars().collect(),
    ])
}

fn chars_to_raw_modes(modes: &[char]) -> Vec<i32> {
    modes.iter().map(|&mode| mode as i32).collect()
}

fn raw_modes_to_cutensor(modes: &[i32]) -> Vec<Mode> {
    modes.iter().copied().map(Mode::new).collect()
}

fn get_extent(modes: &[i32], extent: u64) -> Vec<u64> {
    vec![extent; modes.len()]
}

fn build_distributed_modes(modes: &[i32], value: &str) -> Vec<i32> {
    let cleaned = value.chars().filter(|&c| c != ',').collect::<String>();
    modes
        .iter()
        .copied()
        .filter(|&mode| {
            !cleaned.is_empty() && cleaned.contains(char::from_u32(mode as u32).unwrap_or('\0'))
        })
        .collect()
}

fn get_ranks_per_mode(modes: &[i32], distributed_modes: &[i32], extents: &[u64]) -> Vec<u64> {
    modes
        .iter()
        .zip(extents)
        .map(|(&mode, &extent)| {
            if distributed_modes.contains(&mode) {
                extent
            } else {
                1
            }
        })
        .collect()
}

fn calc_local_extents(global: &[u64], ranks_per_mode: &[u64]) -> Vec<u64> {
    global
        .iter()
        .zip(ranks_per_mode)
        .map(|(&global, &ranks)| ceil_div(global, ranks.max(1)))
        .collect()
}

fn ceil_div(lhs: u64, rhs: u64) -> u64 {
    lhs.div_ceil(rhs)
}

fn product_usize(values: &[u64]) -> Result<usize, Box<dyn std::error::Error>> {
    values.iter().try_fold(1_usize, |product, &value| {
        let value = usize::try_from(value)?;
        product
            .checked_mul(value)
            .ok_or_else(|| "element count overflow".into())
    })
}

fn print_modes(name: &str, values: &[char]) {
    println!(
        "{name}: {}",
        values
            .iter()
            .map(char::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    );
}

fn print_values<T: std::fmt::Display>(name: &str, values: &[T]) {
    println!(
        "{name}: {}",
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    );
}

fn print_performance_metrics(
    modes_a: &[i32],
    modes_b: &[i32],
    modes_c: &[i32],
    extent_a: &[u64],
    extent_b: &[u64],
    extent_c: &[u64],
    min_ms: f64,
) {
    let num_k_modes = (modes_a.len() + modes_b.len() - modes_c.len()) / 2;
    let total_a = product_usize(extent_a).unwrap_or(0);
    let total_b = product_usize(extent_b).unwrap_or(0);
    let total_c = product_usize(extent_c).unwrap_or(0);
    let bytes = (total_a + total_b + total_c) * size_of::<Complex32>();
    let gb_per_rank = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let gflops_per_rank = (2_usize.pow(num_k_modes as u32) as f64 * total_c as f64 * 8.0)
        / (1024.0 * 1024.0 * 1024.0);
    let seconds = min_ms / 1000.0;
    println!(
        "Performance Summary: min_time: {min_ms:.3} ms, data size: ~{gb_per_rank:.3} GB/rank, \
         gflops: {gflops_per_rank:.3} GFLOPS/rank, bandwidth: ~{:.3} GB/rank/s, \
         gflops: ~{:.3} GFLOPS/rank/s",
        gb_per_rank / seconds,
        gflops_per_rank / seconds
    );
}

fn env_parse<T>(name: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    env::var(name).ok()?.parse().ok()
}

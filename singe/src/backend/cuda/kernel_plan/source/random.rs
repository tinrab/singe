use crate::{
    backend::{
        cuda::kernel_plan::source::{KernelInput, KernelPlan, writer::KernelWriter},
        registry::{Operation, RandomAttributes},
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    utility::to_u32,
    value::ValueSpec,
};

/// Philox 4x32-10 and the unit-interval mapping, matching `crate::random`.
const PHILOX: &str = r"
__device__ void philox(unsigned long long seed, unsigned long long counter, unsigned int lanes[4]) {
    unsigned int key0 = static_cast<unsigned int>(seed);
    unsigned int key1 = static_cast<unsigned int>(seed >> 32);
    lanes[0] = static_cast<unsigned int>(counter);
    lanes[1] = static_cast<unsigned int>(counter >> 32);
    lanes[2] = 0;
    lanes[3] = 0;
    for (int round = 0; round < 10; ++round) {
        unsigned long long product0 = 0xD2511F53ULL * lanes[0];
        unsigned long long product1 = 0xCD9E8D57ULL * lanes[2];
        unsigned int next0 = static_cast<unsigned int>(product1 >> 32) ^ lanes[1] ^ key0;
        unsigned int next1 = static_cast<unsigned int>(product1);
        unsigned int next2 = static_cast<unsigned int>(product0 >> 32) ^ lanes[3] ^ key1;
        unsigned int next3 = static_cast<unsigned int>(product0);
        lanes[0] = next0;
        lanes[1] = next1;
        lanes[2] = next2;
        lanes[3] = next3;
        key0 += 0x9E3779B9U;
        key1 += 0xBB67AE85U;
    }
}
__device__ float unit_float(unsigned int bits) {
    return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}
";

/// Plans a random operation whose generator state is its last input, with one thread per sample.
pub fn plan(
    operation: Operation,
    random: &RandomAttributes,
    inputs: &[KernelInput],
) -> Result<KernelPlan> {
    let mut writer = KernelWriter::new();
    let (output, sample) = match operation {
        Operation::Uniform | Operation::Normal => {
            let state = writer.input(DataType::I64, "state")?;
            let output = writer.output(random.data_type, "output")?;
            let (first, second) = (
                format!("{:?}f", random.first),
                format!("{:?}f", random.second),
            );
            let sample = if operation == Operation::Uniform {
                format!(
                    "
philox(seed, offset + index / 4, lanes);
const float value = {first} + ({second} - {first}) * unit_float(lanes[index % 4]);"
                )
            } else {
                format!(
                    "
philox(seed, offset + index / 2, lanes);
const unsigned int pair = static_cast<unsigned int>(index % 2) * 2;
const float radius = sqrtf(-2.0f * logf(1.0f - unit_float(lanes[pair])));
const float value = {first} + {second} * (radius * cosf(6.28318530717958647692f * unit_float(lanes[pair + 1])));"
                )
            };
            let store = match random.data_type {
                DataType::F32 => format!("{output}[index] = value;"),
                DataType::F64 => format!("{output}[index] = static_cast<double>(value);"),
                _ => return Err(Error::unsupported_data_type("cuda random values")),
            };
            (
                (
                    ValueSpec::new(random.data_type, Shape::create(&random.dimensions)?),
                    state,
                ),
                format!("{sample}\n{store}"),
            )
        }
        Operation::Bernoulli => {
            let probabilities = &inputs[0];
            if !matches!(probabilities.data_type, DataType::F32 | DataType::F64) {
                return Err(Error::unsupported_data_type("cuda bernoulli probabilities"));
            }

            let read = writer.input(probabilities.data_type, "probabilities")?;
            let state = writer.input(DataType::I64, "state")?;
            let output = writer.output(DataType::Bool, "output")?;
            (
                (
                    ValueSpec::new(DataType::Bool, probabilities.layout.shape().clone()),
                    state,
                ),
                format!(
                    "
philox(seed, offset + index / 4, lanes);
const float unit = unit_float(lanes[index % 4]);
{output}[index] = unit < {read}[index];"
                ),
            )
        }
        Operation::Multinomial => {
            let weights = &inputs[0];
            let dimensions = weights.layout.shape().dimensions();
            let categories = *dimensions
                .last()
                .ok_or_else(|| Error::missing_input("random plan"))?;
            let ty = match weights.data_type {
                DataType::F32 => "float",
                DataType::F64 => "double",
                _ => return Err(Error::unsupported_data_type("cuda multinomial weights")),
            };
            let read = writer.input(weights.data_type, "weights")?;
            let state = writer.input(DataType::I64, "state")?;
            let output = writer.output(DataType::I64, "output")?;
            let mut shape = vec![random.samples];
            if dimensions.len() == 2 {
                shape.insert(0, dimensions[0]);
            }

            let samples = random.samples;
            (
                (ValueSpec::new(DataType::I64, Shape::create(shape)?), state),
                format!(
                    "
philox(seed, offset + index / 4, lanes);
const {ty}* row = {read} + (index / {samples}ULL) * {categories}ULL;
{ty} total = 0;
for (long long category = 0; category < {categories}; ++category) total = total + row[category];
const {ty} target = static_cast<{ty}>(unit_float(lanes[index % 4])) * total;
{ty} cumulative = 0;
long long chosen = {categories} - 1;
for (long long category = 0; category < {categories}; ++category) {{
    cumulative = cumulative + row[category];
    if (target < cumulative) {{ chosen = category; break; }}
}}
{output}[index] = chosen;"
                ),
            )
        }
        _ => return Err(Error::unsupported_operation("cuda random kernel")),
    };
    let (output, state) = output;
    let block_offset = writer.scalar(
        "block_offset",
        u64::from(to_u32(random.offset, "cuda random offset")?),
        "philox block offset",
    );
    let launch = writer.elements(output.shape().element_count());
    writer.declare(PHILOX);
    writer.lines(&format!(
        "
const unsigned long long seed = static_cast<unsigned long long>({state}[0]);
const unsigned long long offset = static_cast<unsigned long long>({state}[1]) + {block_offset};
unsigned int lanes[4];"
    ));
    writer.lines(&sample);
    Ok(writer.finish(operation.name(), launch, vec![output], inputs, &[operation]))
}

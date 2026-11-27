// Deformable attention: multi-scale value maps sampled bilinearly at learned points and summed by their weights.

use singe_codegen::emit::cuda_c::spelling::{load_expression, store_expression};

use crate::{
    backend::{
        cuda::kernel_plan::source::{KernelInput, KernelPlan, writer::KernelWriter},
        kernel_lowering::element_type,
        registry::Operation,
    },
    data_type::DataType,
    error::{Error, Result},
    shape::Shape,
    utility::to_usize,
    value::ValueSpec,
};

impl KernelPlan {
    pub(crate) fn deformable_attention(inputs: &[KernelInput]) -> Result<Self> {
        let [value, shapes, starts, locations, weights] = inputs else {
            return Err(Error::missing_input("cuda deformable attention operands"));
        };
        if inputs.iter().any(|input| !input.layout.is_contiguous()) {
            return Err(Error::unsupported_operation(
                "cuda deformable attention of strided inputs",
            ));
        }

        let &[batch, rows, heads, channels] = value.layout.shape().dimensions() else {
            return Err(Error::invalid_argument("cuda deformable attention value"));
        };
        let &[_, queries, _, levels, points, _] = locations.layout.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "cuda deformable attention locations",
            ));
        };
        let data_type = value.data_type;
        let element = element_type(data_type)?;
        let ty = if data_type == DataType::F64 {
            "double"
        } else {
            "float"
        };
        let mut writer = KernelWriter::new();
        let values = writer.input(data_type, "values")?;
        let sizes = writer.input(shapes.data_type, "sizes")?;
        let firsts = writer.input(starts.data_type, "firsts")?;
        let places = writer.input(locations.data_type, "places")?;
        let shares = writer.input(weights.data_type, "shares")?;
        let output = writer.output(data_type, "output")?;
        let launch = writer.elements(to_usize(
            batch * queries * heads * channels,
            "cuda deformable attention extent",
        )?);
        let load = |name: &str, input: &KernelInput, index: &str| -> Result<String> {
            Ok(format!(
                "static_cast<{ty}>({})",
                load_expression(element_type(input.data_type)?, &format!("{name}[{index}]"))
            ))
        };
        let load_value = load(
            &values,
            value,
            &format!("((b * {rows}ULL + place) * {heads}ULL + head) * {channels}ULL + channel"),
        )?;
        let load_x = load(&places, locations, "2ULL * sample")?;
        let load_y = load(&places, locations, "2ULL * sample + 1ULL")?;
        let load_weight = load(&shares, weights, "sample")?;
        let stored = store_expression(element, "sum");
        writer.lines(&format!(
            "
const unsigned long long channel = index % {channels}ULL;
const unsigned long long head = (index / {channels}ULL) % {heads}ULL;
const unsigned long long query = (index / ({channels}ULL * {heads}ULL)) % {queries}ULL;
const unsigned long long b = index / ({channels}ULL * {heads}ULL * {queries}ULL);
{ty} sum = 0;
for (unsigned long long level = 0; level < {levels}ULL; ++level) {{
    const long long height = static_cast<long long>({sizes}[2ULL * level]);
    const long long width = static_cast<long long>({sizes}[2ULL * level + 1ULL]);
    const unsigned long long first = static_cast<unsigned long long>({firsts}[level]);
    for (unsigned long long point = 0; point < {points}ULL; ++point) {{
        const unsigned long long sample = (((b * {queries}ULL + query) * {heads}ULL + head) * {levels}ULL + level) * {points}ULL + point;
        const {ty} x = {load_x} * static_cast<{ty}>(width) - static_cast<{ty}>(0.5);
        const {ty} y = {load_y} * static_cast<{ty}>(height) - static_cast<{ty}>(0.5);
        if (!(x > static_cast<{ty}>(-1)) || !(y > static_cast<{ty}>(-1)) || !(x < static_cast<{ty}>(width)) || !(y < static_cast<{ty}>(height))) continue;
        const {ty} weight = {load_weight};
        const {ty} left = floor(x);
        const {ty} top = floor(y);
        const {ty} dx = x - left;
        const {ty} dy = y - top;
        const long long column = static_cast<long long>(left);
        const long long row = static_cast<long long>(top);
        #pragma unroll
        for (int corner = 0; corner < 4; ++corner) {{
            const long long r = row + (corner >> 1);
            const long long c = column + (corner & 1);
            if (r < 0 || c < 0 || r >= height || c >= width) continue;
            const {ty} share = ((corner >> 1) ? dy : static_cast<{ty}>(1) - dy) * ((corner & 1) ? dx : static_cast<{ty}>(1) - dx);
            const unsigned long long place = first + static_cast<unsigned long long>(r * width + c);
            sum += weight * share * {load_value};
        }}
    }}
}}
{output}[index] = {stored};"
        ));
        let shape = Shape::create([batch, queries, heads * channels])?;
        Ok(writer.finish(
            "deformable_attention",
            launch,
            vec![ValueSpec::new(data_type, shape)],
            inputs,
            &[Operation::DeformableAttention],
        ))
    }
}

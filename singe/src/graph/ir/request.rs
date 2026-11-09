use std::collections::HashMap;

use crate::{
    backend::registry::{CheckedOperationRequest, OperationRequest},
    error::{Error, Result},
    graph::ir::{Graph, OperationId},
    value::ValueSpec,
};

impl Graph {
    pub(crate) fn checked_requests(
        &self,
        order: &[OperationId],
    ) -> Result<HashMap<OperationId, CheckedOperationRequest>> {
        let mut requests = HashMap::new();
        for &id in order {
            let operation = self.required_operation(id)?;
            let Some(opcode) = operation.opcode() else {
                continue;
            };
            let inputs = operation
                .inputs()
                .iter()
                .map(|value| {
                    let value = self.required_value(*value)?;
                    Ok(ValueSpec::new(value.data_type(), value.shape().clone()))
                })
                .collect::<Result<Vec<_>>>()?;
            let request =
                OperationRequest::new(opcode, operation.attributes().clone(), inputs).check()?;
            if request.inference().outputs().len() != operation.results().len() {
                return Err(Error::missing_output("checked requests"));
            }

            for (value, expected) in operation.result_values().zip(request.inference().outputs()) {
                let value = self.required_value(value)?;
                if value.data_type() != expected.data_type() {
                    return Err(Error::data_type_mismatch(
                        "prepared operation result",
                        expected.data_type(),
                        value.data_type(),
                    ));
                }
                if value.shape() != expected.shape() {
                    return Err(Error::shape_mismatch(
                        "prepared operation result",
                        expected.shape().dimensions().to_vec(),
                        value.shape().dimensions().to_vec(),
                    ));
                }
            }

            requests.insert(id, request);
        }
        Ok(requests)
    }
}

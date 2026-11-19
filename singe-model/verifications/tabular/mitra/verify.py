"""Mitra, classifying wine and predicting diabetes progression from labeled rows with AutoGluon's `MitraClassifier` and `MitraRegressor`, as `mod.rs` predicts them in Singe.
`transformers` has no tabular pipelines; AutoGluon serves Mitra.
Its defaults fine-tune, mirror features at random, and predict in bfloat16, so the reference turns those off and predicts in float32, the setting Singe's case runs.
The labeled rows are passed as the validation rows too, so the regressor scales its targets by all of them rather than by a random split.
AutoGluon loads the weights inside `fit`, which takes each input's labeled rows, so `run` builds and fits a model for each input: its time includes loading, and the load time reported is near zero."""

import numpy
from autogluon.tabular.models.mitra.sklearn_interface import (
    MitraClassifier,
    MitraRegressor,
)

from support import DEVICE, Verification


def in_precision(model_class, dtype):
    class InPrecision(model_class):
        def _create_config(self, *arguments, **keywords):
            config, model = super()._create_config(*arguments, **keywords)
            config.hyperparams["precision"] = str(dtype).removeprefix("torch.")
            return config, model

    return InPrecision


SETTINGS = {
    "fine_tune": False,
    "random_mirror_x": False,
    "random_mirror_regression": False,
    "device": DEVICE,
    "verbose": False,
}


def tabular_classification(snapshot, dtype):
    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            model = in_precision(MitraClassifier, dtype)(hf_model=snapshot, **SETTINGS)
            rows = numpy.array(item["training_rows"], dtype=numpy.float32)
            labels = numpy.array(item["training_labels"])
            model.fit(rows, labels, X_val=rows, y_val=labels)
            probabilities = model.predict_proba(
                numpy.array(item["rows"], dtype=numpy.float32)
            )
            outputs.append(
                {
                    "labels": probabilities.argmax(axis=1).tolist(),
                    "probabilities": probabilities.tolist(),
                }
            )
        return outputs

    return run


def tabular_regression(snapshot, dtype):
    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            model = in_precision(MitraRegressor, dtype)(hf_model=snapshot, **SETTINGS)
            rows = numpy.array(item["training_rows"], dtype=numpy.float32)
            targets = numpy.array(item["training_targets"], dtype=numpy.float32)
            model.fit(rows, targets, X_val=rows, y_val=targets)
            outputs.append(
                {
                    "predictions": model.predict(
                        numpy.array(item["rows"], dtype=numpy.float32)
                    ).tolist()
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("classification", tabular_classification)
verification.check("regression", tabular_regression)
verification.finish()

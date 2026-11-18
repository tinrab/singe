"""Qwen3, answering decision questions as the decision pipeline of `transformers`' main branch does, and generating greedily after a prompt and after a chat with `transformers`' `generate`, as `verify.rs` generates in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import string  # noqa: E402

import torch  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from support import DEVICE, Verification, generation  # noqa: E402


def text_generation(snapshot, dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        AutoModelForCausalLM.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()
    )

    def run(prompts, parameters):
        outputs = []
        for prompt in prompts:
            # A chat is rendered with the snapshot's template, after which the model answers.
            if isinstance(prompt, list):
                prompt = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **parameters.get("chat_template_kwargs", {}),
                )
            ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    ids, max_new_tokens=parameters["max_new_tokens"], do_sample=False
                )
            outputs.append(generation(model, generated[0, ids.shape[1] :].tolist()))
        return outputs

    return run


verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("chat", text_generation)
verification.check(
    "prompt-bf16", lambda snapshot: text_generation(snapshot, torch.bfloat16)
)


# The generic prompt of `transformers`' decision pipeline, which the installed release predates: `DECISION_PROMPT` of `transformers/decision/template_utils.py`.
DECISION_PROMPT = """
{%- macro text(x) %}{{ "" if x is none else x if x is string else x|tojson }}{% endmacro -%}
{%- set ns = namespace(lines=[text(question.instructions)]) -%}
{%- for option in question.options -%}
    {%- set label, answer = question.labels[loop.index0], text(question.descriptions[loop.index0]) -%}
    {%- if question.type == "noul" -%}
        {%- set answer = answer or (option if question.labels != ["Yes", "No"] else "") -%}
        {%- set ns.lines = ns.lines + ([label ~ ": " ~ answer] if answer else []) -%}
    {%- else -%}
        {%- set answer = (option ~ ": " ~ answer if answer else option) if question.type == "choice" else answer -%}
        {%- set ns.lines = ns.lines + [label ~ ". " ~ answer] -%}
    {%- endif -%}
{%- endfor -%}
{{- text(state) ~ "\\n\\n" ~ ns.lines|select|join("\\n") ~ "\\n" -}}
{%- if question.type == "noul" -%}
    {{- "Answer " ~ question.labels[0] ~ " or " ~ question.labels[1] ~ " only." -}}
{%- else -%}
    {%- set kind = "letter" if question.labels == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: question.labels|length]|list else "label" -%}
    {{- "Answer with the " ~ kind ~ " of the " ~ ("option" if question.type == "choice" else "level") -}}
    {{- " that fits best (" ~ question.labels[0] ~ " to " ~ question.labels[-1] ~ ") only." -}}
{%- endif -%}"""


def parsed(question):
    """A question's options, labels, and descriptions, as `Question.from_request` reads them with the default config."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        criteria = {
            option: (criteria or {}).get(option) for option in ["true", "false"]
        }
    elif kind == "score":
        criteria = {str(index): level for index, level in enumerate(criteria)}
    labels = ["Yes", "No"] if kind == "noul" else list(string.ascii_uppercase)
    options = list(criteria)
    return {
        "type": kind,
        "instructions": question.get("instructions"),
        "options": options,
        "labels": labels[: len(options)],
        "descriptions": list(criteria.values()),
    }


def answer(question, probabilities):
    """The answer from each option's probability, as `Question.answer` writes it."""
    by_option, count = dict(zip(question["options"], probabilities)), len(probabilities)
    if question["type"] == "noul":
        return {"type": "noul", "noul": by_option["true"]}
    best = max(by_option, key=by_option.get)
    if question["type"] == "choice":
        confidence = (
            1.0
            if count == 1
            else max(0.0, (by_option[best] - 1 / count) / (1 - 1 / count))
        )
        return {
            "type": "choice",
            "choice": best,
            "confidence": confidence,
            "probabilities": by_option,
        }
    distance = sum(
        probability * abs(level - int(best))
        for level, probability in enumerate(probabilities)
    )
    uniform = sum(abs(level - (count - 1) / 2) for level in range(count)) / count
    confidence = 1.0 if count == 1 else max(0.0, 1 - distance / uniform)
    score = sum(level * probability for level, probability in enumerate(probabilities))
    return {
        "type": "score",
        "score": score,
        "legend": dict(zip(question["options"], question["descriptions"])),
        "confidence": confidence,
        "probabilities": by_option,
    }


def decision(snapshot):
    """`transformers`' decision pipeline for a model without a decision head: per question, the generic prompt as the user's message of the chat template, one forward pass, and the softmax of the option labels' logits after the prompt."""
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )
    prompt = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string(DECISION_PROMPT)

    def run(requests, parameters):
        outputs = []
        for request in requests:
            answers = {}
            for name, question in request["questions"].items():
                question = parsed(question)
                text = prompt.render(state=request["state"], question=question)
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )["input_ids"].to(DEVICE)
                labels = [
                    tokenizer(label, add_special_tokens=False)["input_ids"][0]
                    for label in question["labels"]
                ]
                with torch.no_grad():
                    logits = model(input_ids=ids).logits[0, -1].float()
                answers[name] = answer(question, logits[labels].softmax(-1).tolist())
            outputs.append(answers)
        return outputs

    return run


verification.check("decision", decision)
verification.finish()

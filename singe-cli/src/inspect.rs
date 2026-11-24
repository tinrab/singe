// `singe inspect` and `singe tasks`: what a model would serve, and which endpoints serve each task.

use std::error::Error;

use anyhow::Result;
use singe_model::{Source, model, task::Task};
use singe_server::registry::endpoints;

pub fn inspect(name: &str) -> Result<()> {
    let source = Source::find(name)?;
    println!("directory: {}", source.directory().display());
    println!("format: {:?}", source.format());
    println!("kind: {}", source.kind());
    let classes = source.classes()?;
    if !classes.is_empty() {
        println!("classes: {}", classes.join(", "));
    }

    match model::default_head(&source) {
        Ok(head) => println!("head: {}", head.id()),
        Err(error) => println!("head: none ({})", report(&error)),
    }

    match Task::default_for(&source) {
        Ok(task) => {
            println!("task: {}", task.id());
            println!("endpoints: {}", endpoints(task.id()).join(", "));
        }
        Err(error) => println!("task: none ({})", report(&error)),
    }

    if let Some(length) = source.context_length()? {
        println!("context length: {length}");
    }

    if let Some(data_type) = source.stored_data_type()? {
        println!("stored data type: {data_type}");
    }
    Ok(())
}

pub fn tasks() {
    for task in Task::ALL {
        if task.heads().is_empty() {
            println!("{}: no pipeline yet", task.id());
        } else {
            println!("{}: {}", task.id(), endpoints(task.id()).join(", "));
        }
    }
}

// An error and its sources on one line.
fn report(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

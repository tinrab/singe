use std::mem;

use crate::{
    backend::cuda::{
        compiled::executable::{CudaCompiledGraph, replay::ReplaySubmission},
        stream::CudaStream,
    },
    error::Result,
};

impl CudaCompiledGraph {
    pub(crate) fn finish_replay_submission(
        &mut self,
        submission: &mut ReplaySubmission,
        stream: &CudaStream,
    ) -> Result<()> {
        if !submission.has_submitted_work() {
            return Ok(());
        }

        let completion = match submission.finish() {
            Ok(completion) => completion,
            Err(error) => {
                stream.synchronize()?;
                submission.clear_retained_values();
                self.record_submission_sync_fallback();
                return Err(error);
            }
        };
        self.pending_runs.push(
            completion,
            mem::take(&mut submission.inputs),
            mem::take(&mut submission.outputs),
        );
        self.replay_stats.completion_events = self.replay_stats.completion_events.saturating_add(1);
        Ok(())
    }

    pub(crate) fn record_replay_success(&mut self, submission: &ReplaySubmission) {
        self.replay_stats.successful_replays =
            self.replay_stats.successful_replays.saturating_add(1);
        self.record_replay_submission_stats(submission);
    }

    pub(crate) fn record_replay_failure(&mut self, submission: &ReplaySubmission) {
        self.replay_stats.failed_replays = self.replay_stats.failed_replays.saturating_add(1);
        self.record_replay_submission_stats(submission);
    }

    fn record_submission_sync_fallback(&mut self) {
        self.replay_stats.submission_sync_fallbacks = self
            .replay_stats
            .submission_sync_fallbacks
            .saturating_add(1);
    }

    fn record_replay_submission_stats(&mut self, submission: &ReplaySubmission) {
        self.replay_stats.input_copies = self
            .replay_stats
            .input_copies
            .saturating_add(submission.input_copies);
        self.replay_stats.output_copies = self
            .replay_stats
            .output_copies
            .saturating_add(submission.output_copies);
        self.replay_stats.native_launches = self
            .replay_stats
            .native_launches
            .saturating_add(submission.native_launches);
        if submission.native_launches == 0 {
            self.replay_stats.metadata_only_replays =
                self.replay_stats.metadata_only_replays.saturating_add(1);
        }
    }
}

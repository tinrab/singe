// ColModernVBert, scoring document pages for text queries, as `verify.py` scores them with `sentence-transformers`.

use singe_model::task::multimodal::visual_document_retrieval::VisualDocumentRetrievalPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<VisualDocumentRetrievalPipeline>("retrieval");
}

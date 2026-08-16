# from sqlalchemy.orm import Session
# from fastapi import Depends
# from app.database import get_db
# from app.services.conversation_manage_service import ConversationService, ConversationNotFoundError
# from app.services.model_collarborate_service import ModelCollaborateService


# class SummarizationService:
#     """
#     Owns summarization logic — formatting a batch of message pairs into a
#     transcript, merging with any prior summary, and persisting the result.
#     Delegates the actual model call to VertexAIService; doesn't talk to
#     Vertex AI directly.
#     """

#     def __init__(
#         self,
#         db: Session = Depends(get_db),
#         conversation_service: ConversationService = Depends(),
#         ai_service: VertexAIService = Depends()
#     ):
#         self.db = db
#         self.conversation_service = conversation_service
#         self.ai_service = ai_service

#     async def summarize_and_append(self, conversation_id: str, pairs: list) -> None:
#         conversation = self.conversation_service.get_conversation_locked(conversation_id)
#         if conversation is None:
#             raise ConversationNotFoundError(conversation_id)

#         transcript = self._format_transcript(pairs)
#         new_chunk = self.ai_service.generate_summary(transcript, previous_summary=conversation.summary)

#         conversation.summary = (
#             f"{conversation.summary}\n{new_chunk}" if conversation.summary else new_chunk
#         )
#         self.db.commit()

#     def _format_transcript(self, pairs: list) -> str:
#         return "\n".join(
#             f"User: {p['user']}\nAssistant: {p['backend']}" for p in pairs
#         )
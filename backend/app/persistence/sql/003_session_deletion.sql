-- Deleting a conversation removes the user's own child rows first
-- (messages, turns, session context) before the session row.
GRANT DELETE ON application.conversation_sessions,
    application.session_contexts,
    application.turns,
    application.conversation_messages TO fredi_app_runtime;

ALTER TABLE application.turns ADD COLUMN replaces_turn_id uuid;
ALTER TABLE application.turns ADD CONSTRAINT turn_revision_target
    FOREIGN KEY (user_id, session_id, replaces_turn_id)
    REFERENCES application.turns (user_id, session_id, turn_id)
    DEFERRABLE INITIALLY DEFERRED;
CREATE UNIQUE INDEX one_completed_revision_per_turn
    ON application.turns (replaces_turn_id)
    WHERE replaces_turn_id IS NOT NULL AND status = 'completed';

-- Failed revisions retain replay status without changing the visible conversation.
CREATE VIEW application.visible_conversation_messages AS
SELECT message.*
FROM application.conversation_messages AS message
LEFT JOIN application.turns AS turn ON turn.turn_id = message.turn_id
WHERE (turn.replaces_turn_id IS NULL OR turn.status = 'completed')
  AND NOT EXISTS (
    SELECT 1 FROM application.turns AS revision
    WHERE revision.replaces_turn_id = message.turn_id AND revision.status = 'completed'
  );
GRANT SELECT ON application.visible_conversation_messages TO fredi_app_runtime;

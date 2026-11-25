-- Supply the same first-message heading for chats created before the UI saved titles.
WITH first_messages AS (
    SELECT DISTINCT ON (m.user_id, m.session_id)
        m.user_id, m.session_id,
        btrim(regexp_replace(m.payload->>'content', '[[:space:]]+', ' ', 'g')) AS summary
    FROM application.visible_conversation_messages m
    JOIN application.conversation_sessions s USING (user_id, session_id)
    WHERE s.title IS NULL AND m.role = 'user'
    ORDER BY m.user_id, m.session_id, m.created_at, m.sequence_no
), headings AS (
    SELECT user_id, session_id,
        left(summary, 48) || CASE WHEN length(summary) > 48 THEN '…' ELSE '' END AS title
    FROM first_messages WHERE summary <> ''
)
UPDATE application.conversation_sessions s
SET title = h.title, payload = jsonb_set(s.payload, '{title}', to_jsonb(h.title))
FROM headings h
WHERE s.user_id = h.user_id AND s.session_id = h.session_id AND s.title IS NULL;

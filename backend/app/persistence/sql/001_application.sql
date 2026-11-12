CREATE TABLE application.identities (
    user_id uuid PRIMARY KEY,
    student_id text NOT NULL UNIQUE CHECK (length(btrim(student_id)) BETWEEN 1 AND 255),
    status text NOT NULL CHECK (status IN ('active','disabled')),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload)='object')
);
CREATE TABLE application.external_identities (
    provider text NOT NULL CHECK (provider IN ('local','moodle','mahara')),
    provider_instance text NOT NULL CHECK (length(btrim(provider_instance)) BETWEEN 1 AND 100),
    external_user_id text NOT NULL CHECK (length(btrim(external_user_id)) BETWEEN 1 AND 255),
    user_id uuid NOT NULL REFERENCES application.identities,
    PRIMARY KEY (provider,provider_instance,external_user_id)
);
CREATE INDEX external_owner ON application.external_identities(user_id);
CREATE TABLE application.user_contexts (
    user_id uuid PRIMARY KEY REFERENCES application.identities,
    version integer NOT NULL CHECK (version>=1),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload)='object')
);
CREATE TABLE application.conversation_sessions (
    session_id uuid PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES application.identities,
    title text,
    status text NOT NULL CHECK (status IN ('active','archived')),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload)='object'),
    UNIQUE(user_id,session_id)
);
CREATE INDEX sessions_by_owner ON application.conversation_sessions(user_id,updated_at DESC,session_id);
CREATE TABLE application.session_contexts (
    session_id uuid PRIMARY KEY,
    user_id uuid NOT NULL,
    version integer NOT NULL CHECK (version>=1),
    updated_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload)='object'),
    FOREIGN KEY(user_id,session_id) REFERENCES application.conversation_sessions(user_id,session_id)
);
CREATE TABLE application.turns (
    turn_id uuid PRIMARY KEY,
    user_id uuid NOT NULL,
    session_id uuid NOT NULL,
    request_id uuid NOT NULL,
    status text NOT NULL CHECK (status IN ('processing','completed','failed','interrupted')),
    request_hash text NOT NULL CHECK (request_hash ~ '^[0-9a-f]{64}$'),
    user_message_id uuid NOT NULL,
    assistant_message_id uuid,
    user_role text NOT NULL DEFAULT 'user' CHECK (user_role='user'),
    assistant_role text NOT NULL DEFAULT 'assistant' CHECK (assistant_role='assistant'),
    error_code text CHECK (error_code ~ '^[a-z][a-z0-9_]{0,63}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(user_id,session_id,request_id),
    UNIQUE(user_id,session_id,turn_id),
    FOREIGN KEY(user_id,session_id) REFERENCES application.conversation_sessions(user_id,session_id),
    CHECK ((status='completed')=(assistant_message_id IS NOT NULL)),
    CHECK ((status IN ('failed','interrupted'))=(error_code IS NOT NULL))
);
CREATE TABLE application.conversation_messages (
    message_id uuid PRIMARY KEY,
    sequence_no bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
    user_id uuid NOT NULL,
    session_id uuid NOT NULL,
    turn_id uuid,
    role text NOT NULL CHECK (role IN ('user','assistant')),
    created_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload)='object'),
    UNIQUE(turn_id,role),
    UNIQUE(user_id,session_id,turn_id,message_id,role),
    FOREIGN KEY(user_id,session_id) REFERENCES application.conversation_sessions(user_id,session_id),
    FOREIGN KEY(user_id,session_id,turn_id) REFERENCES application.turns(user_id,session_id,turn_id)
        DEFERRABLE INITIALLY DEFERRED
);
CREATE UNIQUE INDEX one_processing_turn ON application.turns(user_id,session_id)
    WHERE status='processing';
CREATE INDEX history_by_owner ON application.conversation_messages
    (user_id,session_id,created_at,sequence_no);
ALTER TABLE application.turns ADD CONSTRAINT user_turn_message FOREIGN KEY
    (user_id,session_id,turn_id,user_message_id,user_role)
    REFERENCES application.conversation_messages(user_id,session_id,turn_id,message_id,role)
    DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE application.turns ADD CONSTRAINT assistant_turn_message FOREIGN KEY
    (user_id,session_id,turn_id,assistant_message_id,assistant_role)
    REFERENCES application.conversation_messages(user_id,session_id,turn_id,message_id,role)
    DEFERRABLE INITIALLY DEFERRED;

REVOKE ALL ON ALL TABLES IN SCHEMA application FROM PUBLIC;
GRANT SELECT,INSERT,UPDATE ON application.identities,application.user_contexts,
    application.conversation_sessions,application.session_contexts,application.turns TO fredi_app_runtime;
GRANT SELECT,INSERT ON application.conversation_messages TO fredi_app_runtime;
GRANT SELECT,INSERT,DELETE ON application.external_identities TO fredi_app_runtime;
GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA application TO fredi_app_runtime;

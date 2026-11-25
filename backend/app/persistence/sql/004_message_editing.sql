-- Message text lives in the serialized payload. Keep identity and turn columns read-only.
GRANT UPDATE (payload) ON application.conversation_messages TO fredi_app_runtime;

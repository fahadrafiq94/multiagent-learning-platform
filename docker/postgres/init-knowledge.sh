#!/bin/bash
set -euo pipefail

# This file is sourced by the official Postgres entrypoint on first initialization.
# Keep secrets out of SQL command-line arguments and initialization logs.
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    --no-psqlrc --set ON_ERROR_STOP=1 <<'SQL'
\getenv writer_password KNOWLEDGE_WRITER_PASSWORD
\getenv reader_password KNOWLEDGE_READER_PASSWORD
CREATE ROLE fredi_knowledge_writer LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD :'writer_password';
CREATE ROLE fredi_knowledge_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD :'reader_password';
REVOKE ALL ON DATABASE fredi_knowledge FROM PUBLIC;
GRANT CONNECT ON DATABASE fredi_knowledge TO fredi_knowledge_writer, fredi_knowledge_reader;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
CREATE EXTENSION vector;
CREATE SCHEMA knowledge AUTHORIZATION fredi_knowledge_writer;
GRANT USAGE ON SCHEMA knowledge TO fredi_knowledge_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE fredi_knowledge_writer IN SCHEMA knowledge
    GRANT SELECT ON TABLES TO fredi_knowledge_reader;
SQL

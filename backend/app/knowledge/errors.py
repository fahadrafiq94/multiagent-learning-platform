class KnowledgeError(RuntimeError):
    """Public errors never include database credentials or source contents."""


class KnowledgeConflict(KnowledgeError):
    pass


class KnowledgeIntegrityError(KnowledgeError):
    pass


class KnowledgeNotFound(KnowledgeError):
    pass


class KnowledgePermissionError(KnowledgeError):
    pass


class KnowledgeUnavailable(KnowledgeError):
    pass

"""Contract types and safety helpers shared between harness and agent."""

from dataclasses import dataclass, field

# Modules that train.py is NOT allowed to import.
# The harness checks this via AST inspection before executing.
BANNED_IMPORTS: frozenset[str] = frozenset([
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "imaplib",
    "poplib",
    "ctypes",
    "cffi",
    "mmap",
    "pty",
    "pexpect",
    "paramiko",
    "fabric",
])


@dataclass
class RunRecord:
    """One iteration of the autoresearch loop."""

    iteration: int
    val_ndcg10: float
    model_name: str
    notes: str
    timestamp: str
    train_py_hash: str
    duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "val_ndcg10": self.val_ndcg10,
            "model_name": self.model_name,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "train_py_hash": self.train_py_hash,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(
            iteration=d["iteration"],
            val_ndcg10=d["val_ndcg10"],
            model_name=d.get("model_name", "unknown"),
            notes=d.get("notes", ""),
            timestamp=d["timestamp"],
            train_py_hash=d.get("train_py_hash", ""),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error", ""),
        )

# Security

Do not post customer examples, credentials, model adapters, or database files in public issues. Report a vulnerability privately through this repository's GitHub security reporting when enabled, or contact the maintainer through their published profile.

The initial service runs on loopback, requires a workspace-scoped token and explicit AI-use permission, rejects browser-origin requests, and has no cloud inference fallback. Non-loopback listeners require TLS. Evaluation tokens cannot submit human feedback. Human reviewer tokens are issued separately and bind one reviewer identity; the operator remains responsible for authenticating that human and preventing exposure to AI suggestions.

The application encrypts learning records, cached results and queued receipts. Model files, exported training outputs and local diagnostic reports require an encrypted volume and protected backups. Encryption keys live separately from the encrypted records but on the same machine by default; this does not protect against an administrator controlling a running host. Use a dedicated service account, device encryption and a network egress policy for private installations.

Checkpoint provenance is pinned and hashed; locally signed bundles require the trusted customer key. Signatures prove identity and integrity, not rating accuracy. Public CI processes synthetic fixtures only. Never attach a private-data runner to untrusted pull requests.

Grant withdrawal prevents managed future use and retires affected snapshots and models. It cannot erase knowledge from already copied weights. Replacement training and deletion of controlled copies are required; backup expiration must follow the customer's policy.

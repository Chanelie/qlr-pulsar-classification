# Security policy

This is research software and is not designed for safety-critical deployment.
Do not open a public issue for a suspected security vulnerability. Contact the
repository owner privately and include the affected version, reproduction steps,
and potential impact. Supported security updates apply only to the latest tagged
release.

Never load an untrusted pickle cache. Python pickle files may execute arbitrary
code during deserialization. Load only caches you generated yourself or
obtained from a trusted source, and verify checksums when available.

# Candidate B synthesis

## Base and graft

Candidate B is based on Hermes `1619b409b06ebfab18e9b964ef3e351f82778dcf`
with the wrapped-cause fix from `59acc45225091c984336341bbc5876cd4ec7eec0`.
The implementation uses one bounded cause extractor for the classifier.

The Candidate A graft pattern was the right structural starting point:
return the matched inner typed exception so classification cannot read an
outer SDK wrapper. Candidate A's finite-allowlist match was rejected in the
final adjustment because it changed the base contract for unknown strings.
Candidate B now matches any string subtype exactly as the base did, then
projects only finite values into `error_context`.

## Judge gap and correction

The earlier B implementation used the finite allowlist during recognition.
That incorrectly turned an otherwise typed denial with an unknown string
subtype into a generic error. The corrected path keeps
`provider_relay_denied`, emits a bounded `None` subtype for unknown strings,
and fails typed recognition for non-string or absent values. Downstream
runtime contracts therefore omit invalid subtype fields without changing
auth, retry, fallback, provider, gateway, or live behavior.

## Verification

- Provider-relay classifier focus: 3 passed.
- G1 runtime-process relay focus: 2 passed.
- Ruff, Python compilation, and `git diff --check`: passed.
- No build, provider, or live execution performed.

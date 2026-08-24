# Candidate B: wrapped provider-relay denial cause

## Root cause

`_is_provider_relay_denied_error` already searched the SDK exception cause
chain, but `classify_api_error` read `reason_subreason` from the original
outer exception. A wrapped `ProviderRelayDeniedError` was therefore
recognized as a local relay denial while its finite subtype was lost.

## Chosen change

`_find_provider_relay_denied_error` performs the existing bounded five-level
cause walk and returns the matching typed exception. It preserves the base
typed predicate: any string subtype recognizes the local denial, while
non-string and absent values do not. Classification then applies the existing
finite allowlist helper to the matched subtype, so unknown strings become
`None` and are omitted by downstream runtime contracts.

The focused regression gives the outer wrapper an unrelated subtype to prove
that only the matched inner finite value is carried forward, and covers
unknown, non-string, and absent subtypes. No provider call, retry,
fallback, authentication, gateway, or live-runtime behavior was changed.

## Alternatives rejected

- Reading the outer exception and copying its value: preserves the data-loss
  bug and can trust an unrelated wrapper value.
- Keeping separate cause-chain walks for recognition and extraction: allows
  the two paths to drift and can select different ancestors.
- Passing through arbitrary subtype text or exception messages: violates the
  finite fail-closed contract and risks exposing unrelated/provider data.

## Verification

- Focused provider-relay classifier tests: 3 passed.
- Focused G1 runtime-process relay tests: 2 passed.
- Ruff checks for changed Python files: passed.
- Python compilation for changed Python files: passed.
- Full provider-relay module: 20 passed, 2 pre-existing failures in relay
  header mutation expectations on the untouched 1619 baseline.

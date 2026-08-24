# Candidate B: wrapped provider-relay denial cause

## Root cause

`_is_provider_relay_denied_error` already searched the SDK exception cause
chain, but `classify_api_error` read `reason_subreason` from the original
outer exception. A wrapped `ProviderRelayDeniedError` was therefore
recognized as a local relay denial while its finite subtype was lost.

## Chosen change

`_find_provider_relay_denied_error` performs the existing bounded five-level
cause walk and returns the matching typed exception. It uses the existing
finite allowlist helper while matching, so unknown, non-string, and unrelated
values fail closed. Classification and the retained boolean predicate both
reuse this extractor; classification reads the subtype from the matched
inner exception.

The focused regression also gives the outer wrapper an unrelated subtype to
prove that only the matched inner finite value is carried forward, then
covers unknown, non-string, and absent subtypes. No provider call, retry,
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

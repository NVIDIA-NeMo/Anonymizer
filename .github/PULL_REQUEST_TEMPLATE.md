## Related Issue
<!-- Use "Fixes #123", "Closes #123", or "Resolves #123". External contributors must link a maintainer-triaged issue. -->
<!-- For maintainer-owned changes where no issue is needed, write: "No linked issue required: <reason>". -->

## Plan Document
<!-- For non-trivial changes, link the plan document path or URL. Otherwise write: "No plan required: <reason>". -->

## Summary
<!-- Briefly describe what changed and why. -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring
- [ ] CI, release, or contributor workflow update

## Contributor Checklist
- [ ] PR title follows Conventional Commits, for example `fix: handle empty entity list`
- [ ] Related issue is linked, or a maintainer-owned no-issue reason is documented above
- [ ] For non-trivial changes, a plan document is linked above, or the no-plan reason is documented above
- [ ] Public API impact checked; `skills/anonymizer/SKILL.md` updated if needed
- [ ] No real PII added to tests, docs, notebooks, fixtures, or artifacts
- [ ] No API keys, service tokens, private keys, credentials, or real endpoint secrets added

## Validation
<!-- Choose from the relevant commands below and list what you actually ran. If a relevant check was skipped, explain why. -->
<!-- Common options: make check, make test, make coverage, make test-e2e, make docs-build, make convert-notebooks -->
- Commands run:
- Skipped checks or known failures:

## Documentation and Artifacts
- [ ] Docs updated, or not needed
- [ ] If docs changed: `make docs-build` passes locally
- [ ] If tutorial sources changed: notebooks regenerated with `make convert-notebooks`
- [ ] If e2e, benchmark, or model-provider behavior changed: relevant validation is listed above

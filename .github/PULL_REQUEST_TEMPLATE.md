## Related Issue
<!-- Use "Fixes #123", "Closes #123", or "Resolves #123". -->
<!-- If no issue is needed, write: "No linked issue required: <reason>". -->

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
- [ ] Related issue is linked, or the no-issue reason is documented above
- [ ] For non-trivial changes, a plan PR is linked or not needed
- [ ] Public API impact checked; `skills/anonymizer/SKILL.md` updated if needed
- [ ] No real PII added to tests, docs, notebooks, fixtures, or artifacts

## Testing
- [ ] `make test` passes locally
- [ ] `make check` passes locally
- [ ] Added or updated tests, or explained why tests do not apply

## Documentation and Artifacts
- [ ] Docs updated, or not needed
- [ ] If docs changed: `make docs-build` passes locally
- [ ] If tutorial sources changed: notebooks regenerated with `make convert-notebooks`
- [ ] If e2e, benchmark, or model-provider behavior changed: relevant validation is listed below

## Validation Notes
<!-- Include exact commands, failures you could not resolve, skipped checks, benchmark/e2e notes, or reviewer context. -->

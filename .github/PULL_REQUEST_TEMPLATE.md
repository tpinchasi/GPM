## What this changes, and why

<!-- The why matters more than the what; the diff shows the what. -->

## Checklist

- [ ] Tests that would fail without this change
- [ ] Nothing in the default suite can reach a real provider or need a GPU
- [ ] If this changes *why* something works the way it does, `docs/decisions.md` has a numbered
      entry (what, why, what was rejected) and the relevant `docs/spec/` file is updated
- [ ] Commits are signed off (`git commit -s`) — see CONTRIBUTING.md

### If this touches money or credentials

- [ ] Nothing can rent without a lease, and no lease that can rent exists without a dollar cap
- [ ] Every cap and ceiling is still re-checked by the supervisor *after* the strategy returns
- [ ] The account credential still reaches no host, log, event, API response or browser

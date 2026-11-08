# S14 — A provider account before renting is set up

> Status: **built against fakes** (2026-11-08, D172; gpm-server 0.41.0). Part of the
> [feature list](README.md).

## The story

An operator adds a provider account (Vast.ai, RunPod, …) on the Providers screen the way they add
a hosted API: choose it, name it, type its credential, test it, save. Nothing about what to rent.
Renting is set up separately, on Rented capacity: the disk each machine is rented with, the most
one may cost an hour, the engine rented machines run, and which of the pool's models they are
rented for. Only models that engine can serve are offered. Until renting is set up, the pool rents
nothing.

## Why

Found by the owner adding Vast.ai to a pool that rented nothing (S13's first-account flow):

- **"Why do I need to give disk size and budget? This is only the provider screen."** The first
  account dragged the whole renting setup into the account form, because a `rented:` section was
  not valid without both ceilings (D108).
- **Refused on save:** *rented hosts run 'vllm', and the catalog has no build of ['gemma-4-e4b',
  'speech-he'] for it.* The form defaulted rented machines to vLLM, and "no model ticked" meant
  every model in the pool. The pool's models had builds only for Ollama and whisper.cpp.

## What was built

- **The account alone.** `POST /pool/providers` without `first` writes `rented: {providers: …}`
  and nothing else. The plan says nothing is rented until renting is set up, and asks for nothing
  to be typed again, since such an account spends nothing.
- **"Not set up" is a state.** A `rented:` section whose search in force lacks `min_disk_gb` or
  `max_all_in_hourly` loads. Until both are set:
  - no lease that may rent opens (`LeaseRefused`, saying what is missing);
  - `rent_one` refuses;
  - the worst-case hourly bound is $0;
  - the checks of what rented machines would serve wait, and a pool not set up is not counted as
    holding any model.

  Search profiles still name both ceilings.
- **Set up renting on Rented capacity.** Until renting is set up, the screen shows only a set-up
  panel with four parts:
  - the disk each machine is rented with;
  - the hourly ceiling, all-in;
  - the engine (vLLM or Ollama), each labelled with how many of the pool's models it can serve.
    The one that can serve more is chosen first.
  - the models machines are rented for. Only those the engine has a build of are listed; each other
    is listed with why (no build for it, or something the engine cannot serve).

  Saving goes through validate → plan → apply and is typed again. The renting tabs follow.
- **Changed, never removed.** Once set, the ceilings can be changed but not taken out: the plan
  refuses it. A pool holding machines would have nothing to re-bid them under.
- `GET /pool/renting` and `POST /pool/renting/setup`; `GET /pool/status` gains
  `renting: {set_up, missing}`. The Providers screen links to the set-up while renting is not
  set up.

## What it gives up

- **Still a restart after the first account.** The renting machinery is built when the pool
  starts. Building it live at the first account is separate work.
- **`first` stays on `POST /pool/providers`** (released in 0.40.0) for a caller that wants both in
  one request. The console no longer sends it.

## Tests

`tests/test_renting_setup.py`:
- the account alone, then set-up refused for a model the engine has no build of, then planned,
  typed again and applied;
- a pool not set up opens no renting lease, but opens one that does not rent, and rents once set up
  from the console;
- ceilings cannot be removed once set;
- a half-set policy says what it lacks;
- search profiles still name both;
- a browser run of the set-up panel.

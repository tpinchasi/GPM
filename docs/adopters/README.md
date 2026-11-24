# Writing an adopter guide

An application adopting GPM keeps its own guide in **its own repository**, linking to these docs,
never the other way round. GPM's docs read completely without any adopter's material, and nothing
in the specification, code or identifiers is named after one (CLAUDE.md, "Generic core").

A useful guide holds what is particular to that application:

1. **What it calls today, and what that becomes.** For each place it reaches a model: the
   endpoint, the key and the format (Ollama's API, OpenAI's or Anthropic's), and the one URL and
   key it uses instead ([app-contract.md](../spec/app-contract.md)).
2. **The models it needs, as catalog entries.** Names, builds per engine, kinds (`chat`,
   `embed`, `transcribe`, `rerank`, `predict`), what each build takes in, and any `thinking`
   default ([hosts-routing-capacity.md](../spec/hosts-routing-capacity.md) §4).
3. **Its hosts.** Machines it owns (with agents), fixed remote hosts, hosted APIs, and, if it
   rents, the provider account and the ceilings ([quickstart.md](../quickstart.md),
   [quickstart-renting.md](../quickstart-renting.md)).
4. **Its own workloads, if it wants them.** Latency target, answers at once and budget
   ([workloads.md](../spec/workloads.md)).
5. **What to check after switching.** The same answers through the pool as straight from the
   engine, the waits it tolerates (the SDK waits up to its configured maximum), and the console's
   request log.

Keep real configuration values, prices, measurements and code analysis in the adopter's
repository: they describe that application, not GPM.

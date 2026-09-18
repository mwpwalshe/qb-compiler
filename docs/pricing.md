# Prices: where each number came from, and what it is a number about

A cost estimate is worth what its provenance is worth. Every price this package returns says
whether it is **live**, **cached** or **static**, and the date it was last checked against the
vendor. Three of the vendors below do not sell shots at all, so their per-shot figure is a
conversion or a model with a circuit behind it, and the breakdown names which.

## Live, cached, static

| status | means |
|---|---|
| `live` | fetched from the signed feed in this call, signature verified |
| `cached` | served from a verified feed on disk, younger than the TTL (24 hours by default) |
| `static` | the table that shipped with this package |

**Nothing fetches at import.** Importing `qb_compiler`, building a compiler, or estimating a cost
opens no socket. The feed is read only when it is asked for: `prefer_live=True`, `--live` on the
command line, or `QBC_PRICING_LIVE=1`. One attempt, five second timeout, then the shipped table
with the reason recorded. Offline behaviour is what it always was, plus a status field.

```python
from qb_compiler.cost import get_pricing_provider

provider = get_pricing_provider()                 # static, no network
entry = provider.get("ibm_fez")
breakdown = entry.job_cost(4096)
breakdown.usd                                     # 0.65536
breakdown.status                                  # 'static'
breakdown.assumptions["assumed_shots_per_second"] # 10000, and that assumption is ours
```

```bash
qbc pricing show
qbc pricing show --live --json
qbc when circuit.qasm --live
qbc measure-plan h2.json --backend ibm_fez --live
qbc doctor --live
```

## How the vendors actually bill

| model | who | what it needs |
|---|---|---|
| `per_shot` | Braket devices | shots, and the flat per-task fee |
| `per_second` | IBM | shots, plus an assumed throughput, because IBM publishes seconds |
| `per_gate_shot` | IonQ on Azure | gate counts; without them it falls back to the Braket per-shot price |
| `per_hqc` | Quantinuum | gate and measurement counts; without them it falls back to a per-shot approximation |

IBM publishes Pay-As-You-Go at 96 USD per minute billed per second, which is 1.60 per second, and
publishes no per-shot price. The per-shot numbers here are a conversion at a stated throughput;
a circuit that runs slower than that costs more than the estimate says.

Quantinuum bills in HQCs: `HQC = 5 + C(N1q + 10 N2q + 5 Nm) / 5000` for `C` shots. Give the
counts and you get the vendor's own arithmetic; leave them out and you get a per-shot
approximation whose circuit is named in the entry's notes.

A model that falls back says so in the breakdown, under `fell_back_from` and `fell_back_because`.
No estimate silently invents a circuit.

## The feed, and what a signature buys

The feed is JSON under schema `qb.pricing_feed.v1`:

```json
{
  "schema": "qb.pricing_feed.v1",
  "generated_at": "2026-09-18T06:20:10+00:00",
  "entries": [{"backend": "ibm_fez", "provider": "ibm", "billing": {"model": "per_second", "...": 0},
               "as_of": "2026-09-17", "source": "...", "notes": "..."}],
  "signature": {"alg": "ed25519", "key_id": "<fingerprint>", "sig": "<base64>"}
}
```

The signature covers the canonical JSON of everything except the signature block: sorted keys, no
insignificant whitespace, UTF-8. Reordering the entries or changing one digit changes those bytes
and breaks the signature. The public key ships in the package, so verification needs nothing from
us and no network:

```bash
qbc pricing verify pricing.json          # exit 0 verified, 2 refused
```

**A feed whose signature does not verify is refused, not used with a warning.** The provider falls
back to the shipped table and the refusal is in the status reason, so a caller can tell a price
that was rejected from one that was merely missing. The same applies to a wrong schema, an
unreachable feed, a timeout and a backend the feed does not carry.

Point at a feed with `QBC_PRICING_FEED`, which takes a URL or a local path:

```bash
QBC_PRICING_FEED=./pricing.json QBC_PRICING_LIVE=1 qbc pricing show
```

A verified feed is cached at `~/.qb-compiler/pricing_cache.json` with the time it was fetched, and
served from there until the TTL runs out.

QubitBoost signs and publishes the feed from its own key; the private half never enters this
repository. Until the feed is published at its default URL, `--live` resolves to the shipped table
with the reason "the feed could not be read", which is the correct answer and is what the status
field will say.

## The shipped table

`qb_compiler.cost.pricing` holds the table, `PRICING_AS_OF` the date it was last checked, and
`get_pricing` warns once when that date is more than 90 days old. The warning is doing its job:
treat an estimate from a stale table as indicative and read the vendor page before it decides a
spend.

`cost_per_shot(backend)` and `get_pricing(backend)` are unchanged and still read that table, so
nothing that used them behaves differently. New code that wants the assumptions, the provenance
and the vendor's own billing model uses the provider and the breakdown.

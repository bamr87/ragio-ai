<!-- kit: governance v0.1.0 -->
# 0002 — Missions

- **Status:** proposed
- **Date:** 2026-11-16
- **Proposed by:** Grok Bot (agent), at the maintainer's request
- **Ratified by:**
- **Affects:** constitution §1; PRD.md §1.1; RAGIO-017
- **Supersedes:** none
- **Related:** [PRD.md](../../PRD.md) §1.1

## Context

Constitution §1 was a single mission sentence or short paragraph, not one direction statement with exactly three numbered missions. The fleet rule is that every repo's constitution §1 uses that form, adopted through a numbered amendment the maintainer ratifies by merging. Mission 1 is the maintainer's own wording, adapted to this repo: provide data, tools, resources and transparency for all stakeholders, practitioners in the domain, and the general public. The ambition is the direction, not a claim that the work is already finished.

The earlier §1 text was:

> Ragio-Lab serves a single researcher who wants the whole Radiolab archive as grounded, second-accurate context for a local LLM: every answer cites episode timestamps and can be heard as a clip (PRD § 1).

## Amendment

Retitle constitution §1 **Missions**. Replace the body of §1 with the following. Any invariants or other normative rules that already lived outside §1 are unchanged.

Ragio-Lab's direction is to become the ultimate open home for all things grounded audio knowledge — data, tools, resources and transparency, gathered in one place that anyone can inspect and reuse.

1. **Provide data, tools, resources and transparency for all stakeholders, researchers, audio and journalism professionals, and the general public.** Researchers, listeners, journalists, builders, and the general public are first-class audiences. The aim is a shared, open commons: the same sources, the same tools, and the same trail of evidence for everyone.
2. **Make every episode queryable, reproducible and auditable, and let AI explain it only with citations.** The path starts with the Radiolab archive — local transcripts with second-accurate timestamps, retrieval over the catalog, and answers that can be heard as a clip — and widens from there. Optional AI stays grounded in retrieved audio and transcript rather than replacing them. A fully local runtime remains the default.
3. **Put that archive in every listener's, researcher's and builder's hands.** Serve it through the local RAG engine, the API and the clip tools, so anyone can search, cite and hear the evidence — an open toolkit for working with all things grounded audio knowledge.

These missions set the course; they do not claim the work is finished. They serve the PRD purpose: a fully local, citation-first archive, starting with Radiolab. The earlier mission — serving a researcher who wants the Radiolab archive as grounded, second-accurate context — is absorbed into these three. Provenance, citation and local-first principles are kept. Every feature must trace to them.

_Source: [../PRD.md](../PRD.md) §1.1._

## Consequences

Agents and reviewers judge fit against these three missions before inventing scope, treating them as the direction of the repo rather than a description of what already ships. The same text is added as PRD.md §1.1 so the constitution's `_Source:` line stays accurate. The PRD purpose, goals, feature catalog and invariants outside this addition are unchanged except for a new catalog entry (RAGIO-016) and a regenerated feature-catalog block where the repo already has one. This PR also aligns the README and AGENTS.md one-liners with the missions and links them to constitution §1.

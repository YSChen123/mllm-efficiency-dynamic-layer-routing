# Migration ledger — 2026-11-03

Task 1: complete — local SHA-256 and all 1,145 execution-file hashes verified; source revisions and dataset paths recorded.

Ruling: requested full repository name, `adallava` branch and small `main` paper index — implements the requested structure.

Ruling: preserve executed source and generate relocated runtime copies — keeps auditability; new wrappers require fresh GPU smoke on a new machine.

Ruling: keep LLaVA/CLIP weights used by ROE — deletion would affect an independent running experiment; Ada-specific weights can be removed now.

Ruling: complete raw predictions/telemetry/logs stay in the local archive, compact evidence goes to Git — offline raw reanalysis needs the local archive.

Task 2: final review completed by a fresh independent reviewer. Two important findings and one path-safety finding (regraded important by executor) enter one fix pass.

Final: fixed nested-venv package inheritance — two real venvs verify inherited modules, inherited editable .pth and training-local override precedence, RED→GREEN.

Final: fixed missing summary-error callback — completion-path test reproduces TypeError, then verifies status is written even if optional summary collection fails, RED→GREEN.

Final: fixed unsafe tmux path characters — conservative path character validation rejects parentheses, ampersand, pipe and redirect before creating a runtime, RED→GREEN.

Final: Ruling: fresh package download availability and GPU numerical execution require a new server and new smoke, not a claim derived from CPU checks — no new full install/training/evaluation is run to make the archive; README states this boundary.

Final: Ruling: live dependency protection, repository privacy/heads and deletion require runtime gates — parent performs them and records receipts before claiming completion.

Task 2: complete — Windows and Linux each pass 27 CPU tests. Linux runtime generation, 17 shell scripts, 498 Python files and real model/source symlinks passed; no copied success markers, no GPU jobs, no fresh dependency installation. Exact counts are recorded in provenance/linux-rehearsal.json.

Task 3: complete — private repository created, main/adallava pushed from the Windows client. API privacy/branch SHAs matched; independent clone fsck and 27 tests passed, 1,672 source/script/test files byte-identical. Code commit verified before deletion: b8aff3674bee71499b2fba50dfabb0d05c40d1ed; main a7ed330dba4ac0aa7494c1e225629441243d5868.

Task 4: complete — 72 Ada-only targets deleted after gates. All 349,215 retained files have unchanged identity manifest. Ada source/environments/trained and official Ada checkpoints removed; all datasets and shared LLaVA/CLIP retained. Temporary transfer/rehearsal directory deleted. Project directory allocation decreased 111.81 GiB, including temporary files 112.16 GiB. Local and Git receipts retained.

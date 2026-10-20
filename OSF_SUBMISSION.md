# OSF embargoed pre-registration — submission packet

Everything below is ready to paste. Nothing here needs to be written.

---

## Before you start

Put the full pre-registration text on your clipboard:

    cd ~/Thesis\ 2027 && pbcopy < PREREGISTRATION.md

---

## Click-path

1. Go to https://osf.io/registries/osf/new  (or osf.io → My Registrations → New
   Registration). You do NOT need an existing OSF project — choose **start from
   scratch**.

2. Template: **Open-Ended Registration**.
   Why this one: it is a single narrative field, so your existing document goes
   in whole. The structured "OSF Preregistration" template asks ~10 separate
   questions and would turn a paste job into a 45-minute rewrite. Open-Ended is
   a full, citable registration with a DOI — it is not a lesser record.

3. Fill the fields from the table below.

4. Submit → when asked about privacy, choose **Embargo**, set the date, confirm.

5. Approval: as sole admin you approve your own registration by clicking the
   link in the confirmation email. If you do nothing it **auto-approves after
   48 hours** — so it will lodge either way, but click the link to timestamp it
   today rather than Monday.

---

## Field values

| Field | Value |
|---|---|
| **Title** | Effort-gap: the economic viability of hollowed "ghost" models under audit — pre-registered hypotheses H1–H5 |
| **Description / narrative** | Paste the whole of `PREREGISTRATION.md` (on your clipboard from the step above). |
| **Category** | Project |
| **Subjects** | Computer Science → Artificial Intelligence; (add) Computer Science → Computer and Systems Architecture |
| **Tags** | `model-auditing` `low-rank-compression` `preregistration` `msc-thesis` `llm` `effort-ratio` |
| **License** | CC-BY 4.0 (or "No license" — either is fine; the record is the point, not reuse terms) |
| **Contributors** | Yourself. Add your supervisor as a **read** contributor if you want them on the record as a witness — this is the cheapest way to give the timestamp a second party. |

---

## Embargo date

Recommended: **30 September 2027.**

Reasoning, since this is the one field where a wrong answer is expensive:
ending an embargo early is a normal, supported action (admin approval, done in
minutes). Extending one is not. So set it comfortably past your submission and
viva rather than tight to your ~5-month experimental window — you lose nothing
by over-shooting and lift it the day you submit. OSF's ceiling is 4 years from
creation, so 2027-09-30 is well inside it.

An embargoed registration is **private in content but public in existence**: the
registry shows that a registration exists, its title, its contributors and its
date, with the body withheld until the embargo lifts. That is exactly the
property you want — it timestamps the hypotheses without publishing them.

---

## Why this beats the supervisor email

The email is a claim by an interested party. This is a third-party, immutable,
DOI-bearing record with a date OSF asserts rather than you. The commit hash
`66cc0d8` and the SHA-256 `aee8137…be07df` inside the document bind it to the
private repo: when you later make the repo public, anyone can run
`git show 66cc0d8:proposal.md | shasum -a 256` and confirm the file matches the
one registered today. That chain is what makes the private repo and the public
record work together.

Send the supervisor email anyway. It costs one minute and covers the case where
you need a human witness rather than a machine one.

---

## After it lodges

Come back with the OSF URL/DOI and I will add it to `PREREGISTRATION.md` as a
one-line cross-reference and push — so the repo points at the registry and the
registry points at the repo.

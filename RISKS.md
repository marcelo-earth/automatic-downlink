# Risks & Mitigations

## Risk 1: No triage-specific dataset exists
**Severity:** HIGH
**Impact:** Can't fine-tune specifically on triage classification (CRITICAL/HIGH/SKIP)

**Mitigation:**
- Fine-tune on VRSBench (satellite captioning, 204K samples) — model learns to describe satellite images
- Triage logic lives in the system prompt, not the training data: "Given your description, classify priority as..."
- The VLM's strength is reasoning from descriptions, not memorizing labels
- If time permits, create a small synthetic triage dataset (100-500 samples) using Claude to label SimSat images

**Status:** Mitigated via prompt engineering approach

---

## Risk 2: App must run without debugging (35% of score)
**Severity:** HIGH
**Impact:** Instant disqualification if judges can't run it

**Mitigation:**
- Docker compose with pinned dependencies from day 1
- README with exact steps, copy-paste commands
- Health check endpoints to verify all services are running
- Test on a clean machine before submission
- Include fallback: pre-computed results if model download fails

**Status:** Addressed in architecture, needs continuous testing

---

## Risk 3: Scope creep in 23 days
**Severity:** MEDIUM
**Impact:** Ship incomplete or buggy product

**Mitigation:**
- PRD defines clear MVP vs nice-to-have
- Weekly milestones with go/no-go checkpoints
- Week 3 is for packaging and testing, not new features
- If fine-tuning takes too long, fall back to base model + strong prompt engineering

**Status:** Managed via timeline discipline

---

## Risk 4: Fine-tuning costs exceed Modal free credits ($30)
**Severity:** MEDIUM
**Impact:** Can't complete fine-tuning or need to pay

**Mitigation:**
- LFM2.5-VL-450M is tiny (450M params) — trains fast on H100
- Modal docs say $30 is "enough to run this example end to end"
- Start with small training run (500 samples) to estimate cost before full run
- Fallback: use LoRA (parameter-efficient fine-tuning) to reduce compute
- Fallback 2: use base model without fine-tuning (weaker but functional)

**Status:** Monitor during Week 1

---

## Risk 5: SimSat API is slow/unreliable
**Severity:** MEDIUM
**Impact:** Demo looks bad if images take 30+ seconds to load

**Mitigation:**
- Pre-fetch and cache a diverse set of images for the demo
- Include both cached demo mode and live API mode
- Sentinel-2 is documented as slow — use Mapbox for real-time demo, Sentinel for pre-fetched analysis

**Status:** Needs testing when SimSat is running

---

## Risk 6: Model hallucinations on satellite imagery
**Severity:** LOW-MEDIUM
**Impact:** Model says "flooding detected" when there's no flooding — undermines credibility

**Mitigation:**
- Fine-tuning on satellite-specific data reduces hallucinations
- Triage prompt includes instruction to be conservative: "If uncertain, classify as MEDIUM, not CRITICAL"
- Demo uses curated images where we know ground truth
- Include confidence indicators in output

**Status:** Monitor during model evaluation

---

## Risk 7: Demo video quality
**Severity:** LOW
**Impact:** 20% of score depends on clear communication

**Mitigation:**
- Script the demo before recording
- Architecture diagram ready (ARCHITECTURE.md)
- Practice the narrative: problem → why space → why VLM → architecture → live demo → bandwidth savings
- Record with screen + voiceover, keep under 5 minutes

**Status:** Week 4 task

---

## Contingency Plan

If fine-tuning fails completely:
1. Use base LFM2.5-VL-450M with crafted system prompt
2. Focus on the pipeline, integration, and demo quality
3. Emphasize the architecture and product vision over model performance
4. Still a viable submission — judges value innovation (35%) and demo (20%) over pure model quality

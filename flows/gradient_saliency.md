# Flow: Gradient Saliency & Counterfactual Hints

**When:** After variants are sorted and winner is identified.
**Requires:** GoEmotions available + `target_emotion` set in request.

---

## Entry — `api/agent.py:120`

```
winner_text = f"{winner['headline']}. {winner['body']}"
  │
  ├─ _profile_to_goemotion_label(target_emotion)  api/agent.py:162
  │   Maps EMOTION_PROFILE → primary GoEmotions label
  │   e.g. "urgent" → "nervousness"
  │
  ├─ TARGET_LABELS[goemotion_label]               api/scoring/goemotion_scorer.py:17
  │   Gets integer index into 28-class output
  │   e.g. "nervousness" → 19
  │
  ├─ goemotion.get_model_and_tokenizer()           api/scoring/goemotion_scorer.py:64
  │   Returns raw PyTorch model + tokenizer
  │
  ├─ get_token_saliency(text, idx, model, tok)     api/scoring/saliency.py:14
  │   Returns [(token, saliency_score), ...]       (see below)
  │
  ├─ build_feedback_hint(text, label, pairs, scores) api/scoring/saliency.py:52
  │   Returns Claude-ready feedback block          (see below)
  │
  └─ compute_counterfactual_hint(target, regions,  api/scoring/emotion.py:77
       saliency_hint=saliency_hint)
      Combines region-gap guidance + gradient hint
```

---

## `get_token_saliency` — `api/scoring/saliency.py:14`

Method A from spec (gradient norm per input embedding):

```
text
  └─ tokenizer → input_ids + attention_mask
       │
       ├─ word_embeddings(input_ids) → embeds  [detach, requires_grad=True]
       │
       ├─ model.train()   ← temporarily enables gradient tracking
       │
       ├─ model(inputs_embeds=embeds, ...) → logits → sigmoid
       │
       ├─ loss = -sigmoid[target_emotion_idx]   ← maximise target emotion
       ├─ loss.backward()
       │
       ├─ saliency = embeds.grad.norm(dim=-1)   ← L2 norm per token
       │
       └─ model.eval()    ← restore inference mode

Returns: [(token_str, saliency_float), ...]  in original token order
```

Special tokens filtered: `<s>`, `</s>`, `<pad>`, `Ġ`, `""`

---

## `build_feedback_hint` — `api/scoring/saliency.py:52`

```
saliency_pairs (sorted ascending)
  ├─ hurting = bottom 3 tokens (lowest saliency = pulling AWAY from target)
  └─ helping = top 3 tokens    (highest saliency = driving target emotion)

current_scores = goemotion_scores dict (6 target labels)
predicted_emotion = argmax(current_scores)

Output: structured text block:
  "EMOTION EVALUATION FEEDBACK
   Current copy scored: {predicted_emotion} (score: X.XX)
   Target emotion: {target_emotion} (score: X.XX)

   Tokens pulling AWAY from {target}: [...]
   → Soften, replace, or remove these words.

   Tokens helping {target}: [...]
   → Amplify or build more phrasing around these.

   Rewrite the ad copy to increase [{target}] signal.
   Keep the brand message. Change the framing, not the facts."
```

This block is designed to go verbatim as a prefix into the next Claude generation call.

---

## Region-gap counterfactual — `api/scoring/emotion.py:77`

Runs independently of saliency (always):
```
EMOTION_PROFILES[target] vs region_scores
  → gaps per region (language, visual, prefrontal)
  → regions with gap > threshold → _COPY_GUIDANCE string

_COPY_GUIDANCE at api/scoring/emotion.py:27:
  language:   "use more narrative storytelling, emotional verbs..."
  visual:     "add concrete sensory imagery — texture, colour..."
  prefrontal: "inject decision-driving language: social proof..."
```

When `saliency_hint` is provided, it appends it: `f"{region_hint}\n\n{saliency_hint}"`

---

## Where results surface (winner only)
| Field | Content |
|---|---|
| `winner.saliency_hint` | Raw gradient feedback block (Claude-ready) |
| `winner.counterfactual_hint` | Region gap text + saliency block combined |

---

## To debug
- Saliency returns `[]` → check `goemotion.is_available()` and that `target_emotion` maps via `_profile_to_goemotion_label()`
- `model.train()` causing issues → check `api/scoring/saliency.py:27` — model restored to `.eval()` in `finally`
- Token strings look garbled → `_clean_token()` at `api/scoring/saliency.py:78` strips `Ġ` prefix
- `saliency_hint` not in winner → check `target_idx is not None` guard at `api/agent.py:126`

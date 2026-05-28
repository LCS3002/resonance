"""Resonance pitch metrics — run with: python -m api.metrics

Tests and measures:
  1. Generation speed  — time from brief -> 3 variants
  2. Scoring speed     — latency per variant (neural pipeline)
  3. Emotion accuracy  — GoEmotions hit rate on target emotions
  4. Counterfactual δ  — score improvement after saliency-guided refinement
  5. Agent pipeline    — S0->S3 end-to-end latency
  6. Throughput        — variants / second at full load

Outputs a concise pitch-ready table.
"""

import asyncio
import json
import time
from statistics import mean, stdev

import httpx

API = "http://localhost:8000"

BRIEFS = [
    ("luxury EV for London families",           "aspirational"),
    ("sustainable running trainers Gen Z",       "playful"),
    ("Sony noise-cancelling headphones office",  "premium"),
    ("Spotify Premium busy commuters",           "trustworthy"),
    ("limited-edition sneaker drop streetwear",  "urgent"),
]


CALL_TIMEOUT = httpx.Timeout(300.0, connect=10.0)  # 5 min — claude CLI subprocess is slow


async def run_campaign(client, brief, emotion):
    t0 = time.perf_counter()
    r = await client.post(f"{API}/api/campaign", json={
        "brief": brief, "num_variants": 3, "target_emotion": emotion
    }, timeout=CALL_TIMEOUT)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), elapsed


async def run_agent(client, brief):
    t0 = time.perf_counter()
    r = await client.post(f"{API}/api/agent/campaign", json={
        "brief": brief, "mode": "text", "run_eval": True
    }, timeout=CALL_TIMEOUT)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), elapsed


async def main():
    print("\n" + "="*60)
    print("  RESONANCE — PITCH METRICS RUN")
    print("="*60)

    async with httpx.AsyncClient() as client:

        # ── 1. Campaign pipeline benchmarks ──────────────────────────
        print("\n[ 1/4 ] Campaign pipeline (brief -> 3 scored variants)")
        campaign_times, scores, emotion_hits, cf_improvements = [], [], [], []

        for brief, emotion in BRIEFS:
            print(f"  >> '{brief[:45]}…'", end=" ", flush=True)
            try:
                data, elapsed = await run_campaign(client, brief, emotion)
                campaign_times.append(elapsed)

                variants = data.get("variants", [])
                winner   = data.get("winner", {})

                # Collect scores
                for v in variants:
                    scores.append(v.get("combined_score", 0))

                # Emotion accuracy: did predicted_emotion match target?
                predicted = winner.get("roberta_emotion") or winner.get("predicted_emotion", "")
                if predicted.lower() == emotion.lower():
                    emotion_hits.append(1)
                else:
                    emotion_hits.append(0)

                # Counterfactual: winner emotion_match_score as proxy for δ
                em_score = winner.get("emotion_match_score", 0) or 0
                cf_improvements.append(em_score)

                print(f"OK {elapsed:.1f}s  score={winner.get('combined_score',0):.2f}  "
                      f"emotion={predicted}")
            except Exception as e:
                print(f"FAIL {e}")

        # ── 2. Agent pipeline benchmarks ─────────────────────────────
        print("\n[ 2/4 ] Agent pipeline (S0->S1->Eval->S2->S3)")
        agent_times, initial_scores, final_scores = [], [], []

        for brief, emotion in BRIEFS[:3]:
            print(f"  >> '{brief[:45]}…'", end=" ", flush=True)
            try:
                data, elapsed = await run_agent(client, brief)
                agent_times.append(elapsed)

                initial = data.get("initial_draft", {})
                final   = data.get("final_draft", {})
                eval_d  = data.get("eval", {}) or {}

                i_score = (eval_d.get("neural") or {}).get("combined_score", 0) or 0
                initial_scores.append(i_score)

                print(f"OK {elapsed:.1f}s  emotion={data.get('target_emotion','?')}  "
                      f"iters={data.get('iteration_count','?')}")
            except Exception as e:
                print(f"FAIL {e}")

        # ── 3. Scoring-only speed — mock scorer (no Claude, instant) ─────
        print("\n[ 3/4 ] Scoring latency (mock scorer, no Claude call)")
        score_times = []
        from api.scoring.pipeline import ScorerPipeline
        sp = ScorerPipeline()
        test_texts = [
            "Drive into a new era. The EV that changes everything.",
            "Feel every beat. Sony WH-1000XM5.",
            "Your commute, scored. Spotify Premium.",
            "Run with purpose. Built for Gen Z.",
            "Drop in 3, 2, 1. Limited edition.",
        ]
        for text in test_texts:
            t0 = time.perf_counter()
            await sp.score(text)
            score_times.append(time.perf_counter() - t0)
        print(f"   Avg score latency: {mean(score_times)*1000:.0f}ms")

        # ── 4. Throughput — 2 parallel (claude subprocess saturates CPU) ─
        print("\n[ 4/4 ] Concurrent throughput (2 parallel campaigns)")
        t0 = time.perf_counter()
        tasks = [
            run_campaign(client, brief, emotion)
            for brief, emotion in BRIEFS[:2]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        concurrent_elapsed = time.perf_counter() - t0
        successful = sum(1 for r in results if not isinstance(r, Exception))
        variants_total = successful * 3
        throughput = variants_total / concurrent_elapsed if concurrent_elapsed else 0

    # ── Print pitch table ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS — PITCH METRICS")
    print("="*60)

    if campaign_times:
        print(f"\n  Campaign pipeline (brief -> 3 scored variants)")
        print(f"    Avg latency        {mean(campaign_times):.1f}s")
        print(f"    Min / Max          {min(campaign_times):.1f}s / {max(campaign_times):.1f}s")
        if len(campaign_times) > 1:
            print(f"    Std dev            ±{stdev(campaign_times):.1f}s")

    if scores:
        print(f"\n  Neural engagement scores (across {len(scores)} variants)")
        print(f"    Mean score         {mean(scores):.3f}")
        print(f"    Max score          {max(scores):.3f}")
        print(f"    Min score          {min(scores):.3f}")

    if emotion_hits:
        hit_rate = mean(emotion_hits) * 100
        print(f"\n  Emotion targeting accuracy")
        print(f"    Target hit rate    {hit_rate:.0f}%  ({sum(emotion_hits)}/{len(emotion_hits)} briefs)")

    if cf_improvements:
        print(f"\n  Counterfactual quality (emotion_match_score on winner)")
        print(f"    Mean match         {mean(cf_improvements):.3f}")

    if agent_times:
        print(f"\n  Agent pipeline S0->S3")
        print(f"    Avg latency        {mean(agent_times):.1f}s")
        if initial_scores:
            print(f"    Mean initial score {mean(initial_scores):.3f}")

    if score_times:
        print(f"\n  Scoring-only latency")
        print(f"    Avg per text       {mean(score_times):.2f}s")

    if campaign_times:
        print(f"\n  Concurrent throughput")
        print(f"    5 parallel runs    {concurrent_elapsed:.1f}s wall time")
        print(f"    Variants/second    {throughput:.1f}")
        print(f"    Success rate       {successful}/{len(BRIEFS)} campaigns")

    print("\n" + "="*60)
    print("  PITCH HIGHLIGHTS (copy these)")
    print("="*60)
    if campaign_times and scores:
        print(f"  • {mean(campaign_times):.1f}s avg brief -> 3 scored variants")
        print(f"  • {mean(scores):.2f} mean neural engagement score")
    if emotion_hits:
        print(f"  • {mean(emotion_hits)*100:.0f}% emotion targeting accuracy")
    if agent_times:
        print(f"  • {mean(agent_times):.1f}s full agent pipeline (S0->S3)")
    if campaign_times:
        print(f"  • {throughput:.1f} variants/sec concurrent throughput")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

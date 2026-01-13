# kol_summarize.py
# Read tweets from a CSV (e.g., output.csv), group by username, then generate
# keywords + one-sentence summary via either OpenAI (ChatGPT) or Gemini API.
#
# Usage examples:
#   python kol_summarize.py --csv /mnt/data/output.csv --provider openai --openai-model gpt-5.2
#   python kol_summarize.py --csv /mnt/data/output.csv --provider gemini --gemini-model gemini-2.5-flash
#
# Env vars:
#   OPENAI_API_KEY=...
#   GEMINI_API_KEY=...

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def clamp_text(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


def parse_json_relaxed(text: str) -> Optional[dict]:
    """
    Try to parse JSON from model output, tolerating extra text.
    Strategy:
      1) find the first {...} block and parse
      2) fallback: strict json.loads
    """
    if not text:
        return None
    text = text.strip()

    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to extract a JSON object block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


# -----------------------------
# Prompting
# -----------------------------
SYSTEM_INSTRUCTIONS = """你是一个资深的加密/科技领域社媒分析助手。
你将基于同一位KOL最近发布的多条推文内容，产出：
1) 关键词：5-8个，中文为主；专有名词/项目名可保留英文；不要加#号；用数组返回
2) 一句话总结：1句中文，<=30字，概括该KOL近期关注主题/立场/动态
要求：只基于给定推文，不要编造；信息不足时也要给出尽量稳健的总结。"""

USER_TEMPLATE = """KOL用户名：{username}

推文样本（最近/高互动优先，已去重）：
{tweets}

请严格输出JSON，格式如下：
{{
  "username": "{username}",
  "keywords": ["...", "..."],
  "summary": "..."
}}"""


def build_user_prompt(username: str, tweet_items: List[Dict[str, Any]], max_tweets: int, max_chars_per_tweet: int) -> str:
    lines = []
    for i, t in enumerate(tweet_items[:max_tweets], 1):
        content = clamp_text(safe_str(t.get("content", "")), max_chars_per_tweet)
        ts = safe_str(t.get("tweet_time", ""))
        link = safe_str(t.get("link", ""))
        engagement = t.get("engagement_score", None)
        if engagement is not None:
            lines.append(f"{i}. ({ts}) [eng={engagement}] {content} {link}".strip())
        else:
            lines.append(f"{i}. ({ts}) {content} {link}".strip())
    tweets_block = "\n".join(lines) if lines else "(无)"
    return USER_TEMPLATE.format(username=username, tweets=tweets_block)


# -----------------------------
# Provider clients
# -----------------------------
@dataclass
class ProviderConfig:
    provider: str  # "openai" or "gemini"
    openai_model: str
    gemini_model: str
    temperature: float
    max_retries: int
    retry_backoff_sec: float


def call_openai(prompt: str, cfg: ProviderConfig) -> str:
    """
    OpenAI Python SDK (Responses API).
    Docs: openai-python + Responses API.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: openai. Install with: pip install -U openai"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    # Responses API: client.responses.create(...)
    # Use instructions + input as recommended.
    resp = client.responses.create(
        model=cfg.openai_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
        temperature=cfg.temperature,
    )
    # SDK exposes output_text
    return getattr(resp, "output_text", "") or ""


def call_gemini(prompt: str, cfg: ProviderConfig) -> str:
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: google-genai. Install with: pip install -U google-genai"
        ) from e

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)

    combined = f"""生成风格要求：
- temperature≈{cfg.temperature}
- 只基于给定推文，不要编造
- 严格输出 JSON

{SYSTEM_INSTRUCTIONS}

{prompt}
"""

    resp = client.models.generate_content(
        model=cfg.gemini_model,
        contents=combined,
    )

    return str(getattr(resp, "text", "") or "")



def call_model(prompt: str, cfg: ProviderConfig) -> str:
    last_err = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            if cfg.provider == "openai":
                return call_openai(prompt, cfg)
            if cfg.provider == "gemini":
                return call_gemini(prompt, cfg)
            raise ValueError(f"Unknown provider: {cfg.provider}")
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries:
                sleep_s = cfg.retry_backoff_sec * (2 ** (attempt - 1))
                print(f"[WARN] API call failed (attempt {attempt}/{cfg.max_retries}): {e}\nRetrying in {sleep_s:.1f}s...",
                      file=sys.stderr)
                time.sleep(sleep_s)
            else:
                break
    raise RuntimeError(f"API call failed after {cfg.max_retries} retries: {last_err}") from last_err


# -----------------------------
# Core logic
# -----------------------------
def compute_engagement_score(row: pd.Series) -> int:
    # Weighted sum; adjust as you like
    fav = int(row.get("favorite_count", 0) or 0)
    rt = int(row.get("retweet_count", 0) or 0)
    rep = int(row.get("reply_count", 0) or 0)
    qt = int(row.get("quote_count", 0) or 0)
    bm = int(row.get("bookmark_count", 0) or 0)
    return fav + 2 * rt + rep + 2 * qt + bm


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns in your CSV: username, tweet_time, content, link, bookmark_count, favorite_count, quote_count, reply_count, retweet_count
    required = ["username", "content"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}. Found: {list(df.columns)}")
    # Fill missing optional columns
    for c in ["tweet_time", "link", "bookmark_count", "favorite_count", "quote_count", "reply_count", "retweet_count"]:
        if c not in df.columns:
            df[c] = ""
    df["engagement_score"] = df.apply(compute_engagement_score, axis=1)
    # Basic clean
    df["username"] = df["username"].astype(str).str.strip()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["username"] != ""]
    df = df[df["content"] != ""]
    return df


def dedup_tweets(tweet_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for t in tweet_items:
        key = sha256_text(safe_str(t.get("content", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def summarize_user(username: str, tweet_items: List[Dict[str, Any]], cfg: ProviderConfig,
                   cache_dir: str, max_tweets: int, max_chars_per_tweet: int) -> Dict[str, Any]:
    # Cache key depends on provider+model+content
    cache_key_payload = {
        "provider": cfg.provider,
        "openai_model": cfg.openai_model,
        "gemini_model": cfg.gemini_model,
        "username": username,
        "tweets": [t.get("content", "") for t in tweet_items[:max_tweets]],
    }
    cache_key = sha256_text(json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True))
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    prompt = build_user_prompt(username, tweet_items, max_tweets=max_tweets, max_chars_per_tweet=max_chars_per_tweet)
    raw = call_model(prompt, cfg)
    parsed = parse_json_relaxed(raw)

    # Fallback if JSON parse fails
    if not parsed or "keywords" not in parsed or "summary" not in parsed:
        parsed = {
            "username": username,
            "keywords": [],
            "summary": clamp_text(raw, 120),
            "raw": raw,
            "warning": "Model output was not valid JSON; summary field contains raw output (truncated).",
        }
    else:
        # Post-process keywords
        kws = parsed.get("keywords", [])
        if isinstance(kws, str):
            kws = [k.strip() for k in re.split(r"[;,，、\n]+", kws) if k.strip()]
        if not isinstance(kws, list):
            kws = []
        kws = [clamp_text(str(k).strip(), 30) for k in kws if str(k).strip()]
        kws = kws[:8]
        parsed["keywords"] = kws
        parsed["summary"] = clamp_text(str(parsed.get("summary", "")).strip(), 60)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="output.csv", help="Path to input CSV (default: output.csv)")
    ap.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider")
    ap.add_argument("--openai-model", default="gpt-5.2", help="OpenAI model for Responses API")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tweets", type=int, default=12, help="Max tweets per user to feed into the model")
    ap.add_argument("--max-chars-per-tweet", type=int, default=240, help="Max chars per tweet to feed into the model")
    ap.add_argument("--sort", choices=["engagement", "recent"], default="engagement",
                    help="How to pick top tweets per user")
    ap.add_argument("--cache-dir", default=".kol_cache", help="Cache directory")
    ap.add_argument("--out-csv", default="kol_summary.csv", help="Output CSV path")
    ap.add_argument("--out-json", default="kol_summary.json", help="Output JSON path")
    ap.add_argument("--sleep", type=float, default=0.3, help="Sleep seconds between users to reduce rate spikes")
    args = ap.parse_args()

    cfg = ProviderConfig(
        provider=args.provider,
        openai_model=args.openai_model,
        gemini_model=args.gemini_model,
        temperature=args.temperature,
        max_retries=4,
        retry_backoff_sec=1.2,
    )

    ensure_dir(args.cache_dir)

    df = pd.read_csv(args.csv)
    df = normalize_df(df)

    # Sort within each user to pick representative tweets
    if args.sort == "engagement":
        df = df.sort_values(["username", "engagement_score"], ascending=[True, False])
    else:
        # "recent": best effort sort by tweet_time string; if your tweet_time is always in the same format, it works reasonably.
        df = df.sort_values(["username", "tweet_time"], ascending=[True, False])

    results: List[Dict[str, Any]] = []
    for username, sub in df.groupby("username", sort=False):
        tweet_items = sub.to_dict(orient="records")
        tweet_items = dedup_tweets(tweet_items)

        print(f"[INFO] Summarizing @{username} ({len(tweet_items)} tweets)...")
        try:
            res = summarize_user(
                username=username,
                tweet_items=tweet_items,
                cfg=cfg,
                cache_dir=args.cache_dir,
                max_tweets=args.max_tweets,
                max_chars_per_tweet=args.max_chars_per_tweet,
            )
        except Exception as e:
            res = {
                "username": username,
                "keywords": [],
                "summary": "",
                "error": str(e),
            }

        results.append(res)
        time.sleep(max(0.0, args.sleep))

    # Write outputs
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["username", "keywords", "summary", "error"])
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "username": r.get("username", ""),
                    "keywords": "; ".join(r.get("keywords", []) or []),
                    "summary": r.get("summary", ""),
                    "error": r.get("error", ""),
                }
            )

    print(f"[DONE] Wrote: {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
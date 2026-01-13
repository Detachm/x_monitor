import os
import csv
import time
import random
import argparse
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dateutil import parser as dtparser
from datetime import datetime, timedelta, timezone


API_BASE = "https://api.tweetscout.io/v2"
USER_TWEETS_PATH = "/user-tweets"  # confirmed usable


# ----------------------------
# Rate limiter + retry on 429
# ----------------------------
@dataclass
class RateLimitConfig:
    max_in_flight: int = 5
    min_interval_s: float = 0.25
    max_retries: int = 6
    base_backoff_s: float = 1.0
    max_backoff_s: float = 30.0


class RateLimiter:
    def __init__(self, cfg: RateLimitConfig):
        self.cfg = cfg
        self.sem = threading.Semaphore(cfg.max_in_flight)
        self.lock = threading.Lock()
        self.next_allowed_ts = 0.0

    def _wait_global_interval(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed_ts:
                time.sleep(self.next_allowed_ts - now)
            self.next_allowed_ts = time.time() + self.cfg.min_interval_s

    def post_json(
        self,
        session: requests.Session,
        url: str,
        headers: Dict[str, str],
        body: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        self.sem.acquire()
        try:
            for attempt in range(1, self.cfg.max_retries + 1):
                self._wait_global_interval()
                resp = session.post(url, headers=headers, json=body, timeout=timeout)

                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        sleep_s = min(int(ra), self.cfg.max_backoff_s)
                    else:
                        sleep_s = min(self.cfg.base_backoff_s * (2 ** (attempt - 1)), self.cfg.max_backoff_s)
                    sleep_s += random.uniform(0, 0.3)  # jitter
                    time.sleep(sleep_s)
                    continue

                if 500 <= resp.status_code < 600:
                    sleep_s = min(self.cfg.base_backoff_s * (2 ** (attempt - 1)), self.cfg.max_backoff_s)
                    sleep_s += random.uniform(0, 0.3)
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                return resp.json()

            raise RuntimeError(f"Failed after {self.cfg.max_retries} retries: {url}")
        finally:
            self.sem.release()


# ----------------------------
# Helpers for TweetScout payload
# ----------------------------
def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ApiKey": api_key,
    }


def parse_dt_any(s: str) -> Optional[datetime]:
    """
    Supports:
    - ISO8601
    - Twitter legacy format: "Wed Dec 17 13:16:05 +0000 2025"
    """
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def extract_tweets(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Your sample shows {"tweets":[...]}
    for k in ("tweets", "data", "results", "result"):
        v = payload.get(k)
        if isinstance(v, list):
            return v
    data = payload.get("data")
    if isinstance(data, dict):
        for k in ("tweets", "results", "result"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []


def extract_cursor(payload: Dict[str, Any]) -> Optional[str]:
    # Your request uses "cursor" in payload; response may return cursor/next_cursor
    for k in ("cursor", "next_cursor", "nextCursor", "next"):
        if payload.get(k):
            return str(payload[k])
    meta = payload.get("meta") or {}
    for k in ("cursor", "next_cursor", "nextCursor", "next"):
        if meta.get(k):
            return str(meta[k])
    return None


def get_user_from_tweet(tweet: Dict[str, Any]) -> Dict[str, Any]:
    u = tweet.get("user")
    return u if isinstance(u, dict) else {}


def get_username(tweet: Dict[str, Any], fallback_username: str) -> str:
    u = get_user_from_tweet(tweet)
    return str(u.get("screen_name") or u.get("username") or fallback_username or "")


def get_user_id(tweet: Dict[str, Any]) -> str:
    u = get_user_from_tweet(tweet)
    return str(u.get("id_str") or u.get("id") or "")


def get_tweet_id(tweet: Dict[str, Any]) -> str:
    return str(tweet.get("id_str") or tweet.get("id") or "")


def get_text(tweet: Dict[str, Any]) -> str:
    return str(tweet.get("full_text") or tweet.get("text") or tweet.get("content") or "")


def get_created_at(tweet: Dict[str, Any]) -> str:
    return str(tweet.get("created_at") or tweet.get("createdAt") or "")


def extract_metrics(tweet: Dict[str, Any]) -> Dict[str, int]:
    # TweetScout may not include these; default to 0 if absent
    m = tweet.get("public_metrics") or tweet.get("metrics") or {}

    def g(*keys: str) -> int:
        for k in keys:
            if k in tweet and tweet[k] is not None:
                try:
                    return int(tweet[k])
                except Exception:
                    pass
            if k in m and m[k] is not None:
                try:
                    return int(m[k])
                except Exception:
                    pass
        return 0

    return {
        "bookmark_count": g("bookmark_count", "bookmarks"),
        "favorite_count": g("favorite_count", "like_count", "likes"),
        "quote_count": g("quote_count", "quotes"),
        "reply_count": g("reply_count", "replies"),
        "retweet_count": g("retweet_count", "repost_count", "retweets"),
    }


# ----------------------------
# Core fetch
# ----------------------------
def fetch_user_last_ndays(
    limiter: RateLimiter,
    api_key: str,
    username: Optional[str],
    user_id: Optional[str],
    days: int,
    max_pages: int,
    output_user_id_map: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, str]]]:
    """
    Return:
      - rows for CSV
      - optional (username, resolved_user_id) if found
    """
    if not username and not user_id:
        return [], None

    url = API_BASE.rstrip("/") + USER_TWEETS_PATH
    headers = build_headers(api_key)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    cursor: Optional[str] = None
    rows: List[Dict[str, Any]] = []
    resolved_uid: Optional[str] = None
    resolved_username: Optional[str] = username

    with requests.Session() as session:
        for _ in range(max_pages):
            body: Dict[str, Any] = {}
            # Based on your probe: link works reliably
            if username:
                body["link"] = f"https://x.com/{username}"
            if user_id:
                body["user_id"] = str(user_id)
            if cursor:
                body["cursor"] = cursor

            data = limiter.post_json(session, url, headers, body)
            tweets = extract_tweets(data)
            if not tweets:
                break

            stop = False
            for t in tweets:
                created_at = get_created_at(t)
                dt = parse_dt_any(created_at)

                # resolve user identity from payload
                if not resolved_uid:
                    uid = get_user_id(t)
                    if uid:
                        resolved_uid = uid
                if not resolved_username:
                    resolved_username = get_username(t, "")

                # time filter
                if dt and dt < cutoff:
                    stop = True
                    continue

                u = get_username(t, username or "")
                tid = get_tweet_id(t)
                link = f"https://x.com/{u}/status/{tid}" if (u and tid) else ""

                m = extract_metrics(t)
                rows.append({
                    "username": u,
                    "link": link,
                    "tweet_time": created_at,
                    "content": get_text(t),
                    **m,
                })

            if stop:
                break

            cursor = extract_cursor(data)
            if not cursor:
                break

    # sort desc
    rows.sort(key=lambda r: parse_dt_any(r["tweet_time"]) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    uid_map_item = None
    if output_user_id_map and resolved_username and resolved_uid:
        uid_map_item = (resolved_username, resolved_uid)

    return rows, uid_map_item


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    fields = [
        "username",
        "link",
        "tweet_time",
        "content",
        "bookmark_count",
        "favorite_count",
        "quote_count",
        "reply_count",
        "retweet_count",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_user_id_map(items: List[Tuple[str, str]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["username", "user_id"])
        for u, uid in sorted(set(items)):
            w.writerow([u, uid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("TWEETSCOUT_API_KEY", ""), help="TweetScout ApiKey (or env TWEETSCOUT_API_KEY)")

    ap.add_argument("--usernames", default="", help="Comma-separated usernames, e.g. binancezh,coinbase")
    ap.add_argument("--usernames-file", default="", help="One username per line")
    ap.add_argument("--user-ids", default="", help="Comma-separated user_id values (optional)")
    ap.add_argument("--out", default="tweets.csv")

    ap.add_argument("--days", type=int, default=3, help="Fetch tweets in last N days (e.g. 1,2,3)")
    ap.add_argument("--max-pages", type=int, default=50)

    ap.add_argument("--max-in-flight", type=int, default=5)
    ap.add_argument("--min-interval", type=float, default=0.25)
    ap.add_argument("--max-retries", type=int, default=6)

    ap.add_argument("--dump-user-id-map", action="store_true", help="Also output username->user_id mapping as user_id_map.csv")
    ap.add_argument("--user-id-map-out", default="user_id_map.csv")

    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Missing --api-key or env TWEETSCOUT_API_KEY")

    usernames: List[str] = []
    if args.usernames:
        usernames.extend([u.strip().lstrip("@") for u in args.usernames.split(",") if u.strip()])
    if args.usernames_file:
        with open(args.usernames_file, "r", encoding="utf-8") as f:
            usernames.extend([line.strip().lstrip("@") for line in f if line.strip()])
    usernames = list(dict.fromkeys([u for u in usernames if u]))

    user_ids: List[str] = []
    if args.user_ids:
        user_ids = [x.strip() for x in args.user_ids.split(",") if x.strip()]

    # Build tasks: each task is either (username, None) or (None, user_id)
    tasks: List[Tuple[Optional[str], Optional[str]]] = []
    tasks.extend([(u, None) for u in usernames])
    tasks.extend([(None, uid) for uid in user_ids])

    if not tasks:
        raise SystemExit("No inputs. Use --usernames/--usernames-file or --user-ids")

    cfg = RateLimitConfig(
        max_in_flight=args.max_in_flight,
        min_interval_s=args.min_interval,
        max_retries=args.max_retries,
    )
    limiter = RateLimiter(cfg)

    all_rows: List[Dict[str, Any]] = []
    uid_map: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.max_in_flight) as ex:
        futures = []
        for uname, uid in tasks:
            futures.append(ex.submit(
                fetch_user_last_ndays,
                limiter,
                args.api_key,
                uname,
                uid,
                args.days,
                args.max_pages,
                args.dump_user_id_map,
            ))

        for fut in as_completed(futures):
            rows, map_item = fut.result()
            all_rows.extend(rows)
            if map_item:
                uid_map.append(map_item)

    write_csv(all_rows, args.out)
    print(f"Saved {len(all_rows)} tweets to {args.out}")

    if args.dump_user_id_map and uid_map:
        write_user_id_map(uid_map, args.user_id_map_out)
        print(f"Saved {len(set(uid_map))} username->user_id to {args.user_id_map_out}")


if __name__ == "__main__":
    main()

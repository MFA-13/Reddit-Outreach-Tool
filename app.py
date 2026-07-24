import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import streamlit as st
import requests
import anthropic
import stripe
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


def get_secret(key, default=None):
    """Read config from Streamlit secrets when deployed, else the local environment."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# UI label -> (Reddit native time filter, max age in days or None for the full window).
# Reddit only natively supports week/month/year/all, so the in-between windows
# fetch the next-larger native window and filter locally by each post's age.
TIME_WINDOWS = {
    "Past week":     ("week",  None),
    "Past 2 weeks":  ("month", 14),
    "Past month":    ("month", None),
    "Past 3 months": ("year",  90),
    "Past 6 months": ("year",  180),
    "Past year":     ("year",  None),
    "All time":      ("all",   None),
}

# Fetch a broad pool ranked by upvotes, then re-rank locally by engagement so that
# heavily-discussed posts aren't dropped just because their upvote count is modest.
POOL_SIZE = 100

# How many post variants to generate, each with a distinct title/hook strategy.
NUM_VARIANTS = 3

# User-selectable writing style, low polish -> high polish. Controls capitalization
# and grammar formality only; the universal anti-AI rules apply at every level.
STYLE_LEVELS = {
    "Very casual": "Very informal, like a quick post typed on a phone. Mostly lowercase, lots of sentence fragments, minimal punctuation. Skip capital letters at the start of sentences.",
    "Loose": "Fairly informal. Lowercase sentence starts here and there, frequent fragments and comma splices, light punctuation. Still easy to read.",
    "Casual": "Conversational and relaxed. Normal sentence capitalization, but a few fragments and the occasional comma splice are fine. Sounds like a real person typing without overthinking it.",
    "Lightly casual": "Normal capitalization and mostly correct grammar. You may open with 'and' or 'but' and use an occasional fragment, but keep it tidy.",
    "Polished": "Correct capitalization, grammar, and punctuation throughout. Conversational but clean, like a thoughtful person who proofreads. No lowercase affectation and no intentional errors.",
}

# Matches most emoji + dingbat ranges, used to gauge how emoji-heavy a sub is.
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U00002300-\U000023FF\U0000FE00-\U0000FE0F]"
)

# How heavily to use emojis. Most subreddits read emoji-heavy posts as marketing,
# so the default is None; offer Subtle/Liberal for communities that tolerate them.
EMOJI_LEVELS = {
    "None": "Do not use any emojis at all.",
    "Subtle": "Use at most 1-2 emojis in the entire post, and only where it genuinely feels natural. Never put an emoji in the title.",
    "Liberal": "Use emojis freely where they fit the tone, including the occasional one in a title, but stop short of looking like spam.",
}

# Reddit blocks unauthenticated scraping at the IP level, so we delegate the
# actual scrape to an Apify actor. Apify handles proxies/credentials and returns
# clean JSON. Token: console.apify.com -> Settings -> API & Integrations.
APIFY_API_TOKEN = get_secret("APIFY_API_TOKEN")
# Default actor: practicaltools/apify-reddit-api. It uses Reddit's API rather than
# scraping HTML, so it avoids the 403 blocks that kill the HTML-scraper actors.
APIFY_ACTOR_ID = get_secret("APIFY_ACTOR_ID", "practicaltools~apify-reddit-api")

# Supabase (auth + usage tracking) and Stripe (billing) credentials.
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
STRIPE_SECRET_KEY = get_secret("STRIPE_SECRET_KEY")
STRIPE_PAYMENT_LINK = get_secret("STRIPE_PAYMENT_LINK")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))


@st.cache_resource
def get_supabase():
    """Cached Supabase client (None if credentials are missing)."""
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        return None
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _first(d, *keys, default=None):
    """Return the first present, non-None value among keys (actors vary on naming)."""
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def _parse_created(value):
    """Parse the actor's ISO 8601 createdAt (e.g. 2026-06-05T16:40:56.000Z)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _detect_media(item):
    """Best-effort guess at whether a post has an image/video attached.

    We never display the media; this only feeds the playbook's format analysis
    (e.g. "the top posts attach a demo video"). isVideo is the strongest signal.
    """
    if item.get("isVideo"):
        return "video"
    url = (item.get("url") or "").lower()
    if any(s in url for s in ("v.redd.it", "youtube.com", "youtu.be")):
        return "video"
    if "reddit.com/gallery" in url:
        return "image gallery"
    if "i.redd.it" in url or "imgur" in url or re.search(r"\.(jpg|jpeg|png|gif|webp)(\?|$)", url):
        return "image"
    html = item.get("html") or ""
    if re.search(r"(i|v)\.redd\.it", html) or re.search(r"<(img|video)\b", html, re.I):
        return "image/video"
    return "none"


def clean_subreddit(name):
    """Normalize user input to a bare subreddit name; reject anything invalid.

    A stray space (or other illegal char) produces a malformed start URL that
    makes the Apify actor fail with a 400, so we catch it here with a clear message.
    """
    s = (name or "").strip()
    if "reddit.com" in s:  # tolerate a pasted full URL
        s = s.split("/r/")[-1]
    s = s.strip().strip("/")
    if s.lower().startswith("r/"):  # tolerate an "r/" prefix
        s = s[2:]
    s = s.strip().strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,21}", s):
        raise RuntimeError(
            f"'{name}' isn't a valid subreddit name. Use just the name — letters, "
            "numbers, and underscores only, no spaces (e.g. applewatch)."
        )
    return s


def _build_post(item, subreddit_name, cutoff=None):
    """Parse one actor item into our post dict, or return None to skip it.

    Shared by fetch_posts (winners) and fetch_losers (baseline) so the parsing
    stays in one place.
    """
    data_type = (item.get("dataType") or "").lower()
    if data_type and data_type != "post":  # skip comments/community items
        return None
    if item.get("isAd"):  # drop promoted posts; we want organic engagement
        return None
    title = _first(item, "title")
    if not title:
        return None

    created = _parse_created(item.get("createdAt"))
    if cutoff and created and created < cutoff:
        return None

    body = _first(item, "body", "text", "selftext", "postText", default="") or ""
    if body in ("[deleted]", "[removed]"):
        body = ""

    score = _first(item, "upVotes", "score", "upvotes", default=0) or 0
    comments = _first(item, "numberOfComments", "num_comments", "comments", default=0) or 0
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    try:
        comments = int(comments)
    except (TypeError, ValueError):
        comments = 0
    ratio = _first(item, "upVoteRatio", "upvoteRatio", "upvote_ratio")
    # Build a stable Reddit thread permalink so the UI can link to each post and
    # so comment mining can target the exact threads.
    parsed_id = item.get("parsedId")
    parsed_comm = item.get("parsedCommunityName") or subreddit_name
    permalink = (
        f"https://www.reddit.com/r/{parsed_comm}/comments/{parsed_id}/"
        if parsed_id else _first(item, "url", default="")
    )
    return {
        "title": title,
        "url": permalink,
        "permalink": permalink,
        "score": score,
        "upvote_ratio": ratio,  # may be None; build_posts_text handles that
        "comments": comments,
        # Blended engagement: an upvote and a comment count equally.
        "engagement": score + comments,
        "flair": _first(item, "flair", "link_flair_text", "linkFlairText") or "none",
        "media": _detect_media(item),
        "created_at": item.get("createdAt"),   # raw ISO; parse on demand
        "scraped_at": item.get("scrapedAt"),    # raw ISO; for age/velocity
        "body": str(body)[:600],
    }


def _apify_error_message(response, subreddit_name):
    """Turn an Apify error response into a clear, actionable message.

    The run-sync endpoint returns 400 both for bad input AND when the actor run
    itself fails (a private/nonexistent/banned subreddit), so we read the
    structured `error.type` to tell them apart instead of guessing.
    """
    if response.status_code == 401:
        return "Apify rejected the token (401). Check APIFY_API_TOKEN."
    if response.status_code == 404:
        return (f"Apify actor '{APIFY_ACTOR_ID}' not found (404). "
                "Set APIFY_ACTOR_ID to a valid actor.")
    if response.status_code == 429:
        return "Apify is rate-limiting you (429). Wait a moment and try again."
    if response.status_code >= 500:
        return f"Apify had a server error ({response.status_code}). Try again shortly."

    err_type = err_msg = ""
    try:
        err = (response.json() or {}).get("error") or {}
        err_type = (err.get("type") or "").lower()
        err_msg = err.get("message") or ""
    except ValueError:
        pass

    if err_type == "run-failed":
        return (f"Couldn't scrape r/{subreddit_name}. This usually means the subreddit "
                "doesn't exist, is private, or is banned/quarantined. Double-check the "
                "spelling — and if it's definitely public, try again, since Apify runs "
                "occasionally fail transiently.")
    if err_type in ("invalid-input", "invalid-request"):
        return f"Apify rejected the request: {err_msg or 'invalid input'}."
    return f"Apify request failed ({response.status_code}). {err_msg or response.text[:150]}"


def fetch_posts(subreddit_name, time_window, limit=50):
    if not APIFY_API_TOKEN:
        raise RuntimeError(
            "APIFY_API_TOKEN is not set. Get one at console.apify.com "
            "(Settings -> API & Integrations) and add it to your .env."
        )

    subreddit_name = clean_subreddit(subreddit_name)
    native_time, max_age_days = TIME_WINDOWS.get(time_window, ("month", None))

    # Run the actor synchronously and get its dataset items back in one call.
    # Token goes in the Authorization header (not the URL) so it can't leak into
    # error messages.
    endpoint = (
        f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
    )
    headers = {"Authorization": f"Bearer {APIFY_API_TOKEN}"}
    actor_input = {
        "startUrls": [{"url": f"https://www.reddit.com/r/{subreddit_name}/"}],
        "sort": "top",
        "time": native_time,
        "maxItems": POOL_SIZE,
        "skipComments": True,
        "skipUserPosts": True,
        "skipCommunity": True,
        "searchPosts": True,
        "searchComments": False,
    }

    try:
        response = requests.post(endpoint, json=actor_input, headers=headers, timeout=300)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Apify timed out (the run took over 5 minutes). Try again, or pick a "
            "smaller time window."
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Couldn't reach Apify. Check your connection. ({e})")
    if not response.ok:
        raise RuntimeError(_apify_error_message(response, subreddit_name))

    # For the non-native windows (e.g. "Past 2 weeks") narrow the larger native
    # window down to posts newer than this cutoff.
    cutoff = None
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    items = response.json()
    posts = [p for p in (_build_post(it, subreddit_name, cutoff) for it in items) if p]

    if not posts:
        raise RuntimeError(
            f"Apify returned no posts for r/{subreddit_name}. The subreddit may be "
            "empty/private, or the actor's output fields differ from what's mapped."
        )

    # Re-rank the upvote-sorted pool by blended engagement so heavily-discussed
    # posts surface even when their raw upvote count is modest.
    posts.sort(key=lambda p: p["engagement"], reverse=True)
    return posts[:limit]


def build_posts_text(posts):
    lines = []
    for i, p in enumerate(posts, 1):
        ratio = p.get("upvote_ratio")
        upvoted = f"{int(ratio * 100)}% upvoted | " if ratio is not None else ""
        lines.append(
            f"{i}. [{p['score']} pts | {upvoted}{p['comments']} comments | flair: {p['flair']} | attached media: {p['media']}]\n"
            f"   Title: {p['title']}\n"
            f"   Body: {p['body'] or '(link post / no body)'}"
        )
    return "\n\n".join(lines)


def recommend_settings(posts):
    """Heuristically recommend a writing style + emoji level from the scraped posts,
    so the UI can default the sliders to match the subreddit's actual norm."""
    default = {"style": "Casual", "emojis": "None", "emoji_ratio": 0.0, "lower_ratio": 0.0}
    if not posts:
        return default

    # Emoji norm: share of posts whose title or body contains an emoji.
    emoji_posts = sum(
        1 for p in posts
        if EMOJI_RE.search(p.get("title", "")) or EMOJI_RE.search(p.get("body", ""))
    )
    emoji_ratio = emoji_posts / len(posts)
    if emoji_ratio >= 0.45:
        emojis = "Liberal"
    elif emoji_ratio >= 0.15:
        emojis = "Subtle"
    else:
        emojis = "None"

    # Casualness proxy: share of titles that start with a lowercase letter.
    lower = counted = 0
    for p in posts:
        first = next((c for c in p.get("title", "") if c.isalpha()), "")
        if first:
            counted += 1
            lower += first.islower()
    lower_ratio = lower / counted if counted else 0.0
    if lower_ratio >= 0.50:
        style = "Very casual"
    elif lower_ratio >= 0.30:
        style = "Loose"
    elif lower_ratio >= 0.12:
        style = "Casual"
    elif lower_ratio >= 0.03:
        style = "Lightly casual"
    else:
        style = "Polished"

    return {"style": style, "emojis": emojis,
            "emoji_ratio": emoji_ratio, "lower_ratio": lower_ratio}


def fetch_subreddit_info(subreddit_name):
    """Best-effort fetch of the subreddit's sidebar description via Apify.

    This is the public sidebar text, NOT the formal numbered rules list (which would
    need authenticated Reddit API access). Returns '' if unavailable.
    """
    if not APIFY_API_TOKEN:
        return ""
    subreddit_name = clean_subreddit(subreddit_name)
    endpoint = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
    headers = {"Authorization": f"Bearer {APIFY_API_TOKEN}"}
    actor_input = {
        "startUrls": [{"url": f"https://www.reddit.com/r/{subreddit_name}/"}],
        "maxItems": 3,
        "skipComments": True,
        "skipUserPosts": True,
        "skipCommunity": False,
        "searchPosts": False,
        "searchCommunities": True,
    }
    try:
        resp = requests.post(endpoint, json=actor_input, headers=headers, timeout=120)
        if not resp.ok:
            return ""
        for item in resp.json():
            if item.get("dataType") == "community":
                return (item.get("description") or "").strip()
    except requests.RequestException:
        return ""
    return ""


DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def analyze_timing(posts):
    """Posting-time + velocity signal from createdAt/scrapedAt (FREE, pure Python).

    velocity = score / age_hours separates posts that earned engagement fast from
    posts that merely sat and accumulated. Directional only (see UI caveat).
    """
    rows = []
    for i, p in enumerate(posts):
        created = _parse_created(p.get("created_at"))
        scraped = _parse_created(p.get("scraped_at"))
        if not created or not scraped:
            continue
        age_h = max((scraped - created).total_seconds() / 3600, 1.0)
        rows.append({
            "idx": i, "age_hours": age_h, "velocity": p["score"] / age_h,
            "dow": created.weekday(), "hour": created.hour,
            "engagement": p["engagement"],
        })
    # Need timestamps on most posts to say anything honest.
    if not posts or len(rows) < max(5, int(0.6 * len(posts))):
        return {"usable": False}

    median_age = statistics.median(r["age_hours"] for r in rows)
    vels = sorted(r["velocity"] for r in rows)
    q75 = vels[int(0.75 * (len(vels) - 1))]
    outperformers = [r["idx"] for r in rows
                     if r["velocity"] >= q75 and r["age_hours"] < median_age]

    dow_v, hour_v = defaultdict(list), defaultdict(list)
    for r in rows:
        dow_v[r["dow"]].append(r["velocity"])
        hour_v[r["hour"]].append(r["velocity"])
    best_dow = sorted(((DOW_NAMES[d], sum(v) / len(v), len(v)) for d, v in dow_v.items()),
                      key=lambda x: x[1], reverse=True)[:3]
    best_hours = sorted(((h, sum(v) / len(v), len(v)) for h, v in hour_v.items()),
                        key=lambda x: x[1], reverse=True)[:3]
    return {"usable": True, "rows": rows, "median_age_hours": median_age,
            "best_dow": best_dow, "best_hours": best_hours,
            "outperformers": outperformers}


# Title features measured for the quantified-patterns edge.
FEATURE_LABELS = {
    "has_number": "number in title",
    "has_currency": "$ amount in title",
    "has_question": "question (?) title",
    "has_emoji": "emoji in title",
    "starts_lower": "lowercase start",
    "has_exclaim": "exclamation (!)",
    "has_media": "image/video attached",
}


def _title_features(p):
    t = p.get("title", "")
    first_alpha = next((c for c in t if c.isalpha()), "")
    return {
        "has_number": bool(re.search(r"\d", t)),
        "has_currency": "$" in t,
        "has_question": "?" in t,
        "has_emoji": bool(EMOJI_RE.search(t)),
        "starts_lower": first_alpha.islower(),
        "has_exclaim": "!" in t,
        "has_media": p.get("media", "none") != "none",
        "word_count": len(t.split()),
    }


def analyze_patterns(posts):
    """Which title features correlate with engagement in THIS sub (FREE, pure Python).

    Directional only: gate each driver on n>=5 per subgroup so 1-post artifacts
    don't show up. n=50 is small; never present as causal.
    """
    if not posts:
        return {"n": 0, "boolean_drivers": [], "word_count": None}
    feats = [(_title_features(p), p["engagement"]) for p in posts]
    drivers = []
    for key, label in FEATURE_LABELS.items():
        withs = [e for f, e in feats if f[key]]
        withouts = [e for f, e in feats if not f[key]]
        if len(withs) >= 5 and len(withouts) >= 5:
            wm, om = sum(withs) / len(withs), sum(withouts) / len(withouts)
            lift = (wm - om) / om * 100 if om else 0.0
            drivers.append({"feature": label, "with_mean": wm, "without_mean": om,
                            "lift_pct": lift, "n_with": len(withs),
                            "n_without": len(withouts)})
    drivers.sort(key=lambda d: abs(d["lift_pct"]), reverse=True)

    wcs = [f["word_count"] for f, _ in feats]
    med = statistics.median(wcs)
    short = [e for f, e in feats if f["word_count"] <= med]
    long_ = [e for f, e in feats if f["word_count"] > med]
    word_count = None
    if len(short) >= 5 and len(long_) >= 5:
        word_count = {"split": med, "short_mean": sum(short) / len(short),
                      "long_mean": sum(long_) / len(long_)}
    return {"n": len(posts), "boolean_drivers": drivers, "word_count": word_count}


def format_drivers(patterns):
    """Compact driver summary for prompts (empty string if nothing solid)."""
    if not patterns or not patterns.get("boolean_drivers"):
        return ""
    lines = []
    for d in patterns["boolean_drivers"][:5]:
        sign = "+" if d["lift_pct"] >= 0 else ""
        lines.append(f"- {d['feature']}: {sign}{d['lift_pct']:.0f}% engagement "
                     f"(n={d['n_with']} with / {d['n_without']} without)")
    wc = patterns.get("word_count")
    if wc:
        better = "shorter" if wc["short_mean"] > wc["long_mean"] else "longer"
        lines.append(f"- title length: {better} titles do better "
                     f"(median split at {wc['split']:.0f} words)")
    return "\n".join(lines)


def fetch_comments(posts, top_n=10, per_post=30, max_total=150):
    """Fetch top comments for the top_n posts (one extra Apify run).

    The actor returns one item per thread, shaped {post:{...}, comments:[...]},
    where each comment has body + upVotes (a string). `maxItems` here is per-post
    and the actor caps it at 100. Returns a flat list of the highest-upvoted
    comments, capped at max_total to bound token cost.
    """
    if not APIFY_API_TOKEN:
        return []
    targets = [p for p in posts[:top_n] if p.get("permalink")]
    if not targets:
        return []
    endpoint = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
    headers = {"Authorization": f"Bearer {APIFY_API_TOKEN}"}
    actor_input = {
        "startUrls": [{"url": p["permalink"]} for p in targets],
        "skipComments": False,
        "searchComments": True,
        "skipUserPosts": True,
        "skipCommunity": True,
        "searchPosts": False,
        "maxItems": min(per_post, 100),  # actor hard-caps maxItems at 100
    }
    try:
        resp = requests.post(endpoint, json=actor_input, headers=headers, timeout=300)
        if not resp.ok:
            return []
        items = resp.json()
    except requests.RequestException:
        return []

    out = []
    for item in items:
        title = (item.get("post") or {}).get("title") or ""
        for c in (item.get("comments") or []):
            body = (c.get("body") or "").strip()
            if not body or body in ("[deleted]", "[removed]"):
                continue
            try:
                ups = int(c.get("upVotes") or 0)
            except (TypeError, ValueError):
                ups = 0
            out.append({"title": title, "body": body[:300], "upvotes": ups})
    out.sort(key=lambda c: c["upvotes"], reverse=True)
    return out[:max_total]


def analyze_audience(comments, subreddit_name):
    """One Claude pass turning raw comments into audience insights. Returns '' if empty."""
    if not comments:
        return ""
    text = "\n".join(f"[{c['upvotes']} pts] {c['body']}" for c in comments)
    prompt = f"""These are top comments from high-performing posts in r/{subreddit_name}.
Extract what this community actually cares about. Be specific and quote real phrases.

{text}

Return exactly these four labeled sections, each a few tight bullet points:
PAIN POINTS: <recurring frustrations or struggles people mention>
RECURRING QUESTIONS: <questions people keep asking>
OBJECTIONS: <skepticism or pushback that shows up>
VOCABULARY: <exact words and phrases the community uses>"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def fetch_losers(subreddit_name, time_window, limit=25):
    """Fetch a lower-engagement baseline for contrast (one extra Apify run).

    Uses sort='new', then keeps only matured posts (age >= 24h) in the bottom
    engagement quartile, so we contrast genuine under-performers, not merely-new posts.
    """
    if not APIFY_API_TOKEN:
        return []
    subreddit_name = clean_subreddit(subreddit_name)
    native_time, _ = TIME_WINDOWS.get(time_window, ("month", None))
    endpoint = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"
    headers = {"Authorization": f"Bearer {APIFY_API_TOKEN}"}
    actor_input = {
        "startUrls": [{"url": f"https://www.reddit.com/r/{subreddit_name}/"}],
        "sort": "new",
        "time": native_time,
        "maxItems": POOL_SIZE,
        "skipComments": True,
        "skipUserPosts": True,
        "skipCommunity": True,
        "searchPosts": True,
        "searchComments": False,
    }
    try:
        resp = requests.post(endpoint, json=actor_input, headers=headers, timeout=300)
        if not resp.ok:
            return []
        items = resp.json()
    except requests.RequestException:
        return []

    now = datetime.now(timezone.utc)
    mature = []
    for it in items:
        p = _build_post(it, subreddit_name)
        if not p:
            continue
        created = _parse_created(p.get("created_at"))
        if created and (now - created).total_seconds() / 3600 >= 24:
            mature.append(p)
    if len(mature) < 8:  # too few to form a meaningful baseline
        return []
    cutoff_eng = statistics.quantiles([p["engagement"] for p in mature], n=4)[0]
    losers = sorted((p for p in mature if p["engagement"] <= cutoff_eng),
                    key=lambda p: p["engagement"])
    return losers[:limit]


def analyze_contrast(winners, losers):
    """Pure-Python: title features that over-index in losers vs winners (>=20pp gap)."""
    if not winners or not losers:
        return None

    def rates(posts):
        agg = {k: 0 for k in FEATURE_LABELS}
        for p in posts:
            f = _title_features(p)
            for k in FEATURE_LABELS:
                agg[k] += 1 if f[k] else 0
        return {k: agg[k] / len(posts) for k in FEATURE_LABELS}

    wr, lr = rates(winners), rates(losers)
    loser_traits, winner_traits = [], []
    for k, label in FEATURE_LABELS.items():
        diff = lr[k] - wr[k]
        if diff >= 0.20:
            loser_traits.append(f"{label} ({lr[k]*100:.0f}% of losers vs {wr[k]*100:.0f}% of winners)")
        elif diff <= -0.20:
            winner_traits.append(f"{label} ({wr[k]*100:.0f}% of winners vs {lr[k]*100:.0f}% of losers)")
    return {"loser_traits": loser_traits, "winner_traits": winner_traits}


def generate_playbook(posts, subreddit_name, timing=None, patterns=None,
                      audience="", contrast=None):
    posts_text = build_posts_text(posts)

    # Optional analysis blocks — only present when their feature ran. Each carries
    # a directional-signal caveat so the playbook never overclaims.
    extra = ""
    drivers_text = format_drivers(patterns)
    if drivers_text:
        extra += (
            "\n\nMEASURED TITLE DRIVERS (computed across the posts above; directional "
            f"signal from a small sample of {patterns['n']}, NOT proof of causation):\n"
            f"{drivers_text}\n"
            "Weave these in only where they agree with what you see in the posts."
        )
    if timing and timing.get("usable"):
        days = ", ".join(f"{d}" for d, _, _ in timing["best_dow"])
        hrs = ", ".join(f"{h:02d}:00 UTC" for h, _, _ in timing["best_hours"])
        extra += (
            "\n\nTIMING SIGNAL (by engagement velocity = score/age; directional, since "
            "Reddit votes plateau — treat as a rough hint, not a guaranteed best time):\n"
            f"- Highest-velocity days: {days}\n- Highest-velocity hours: {hrs}\n"
            f"- {len(timing['outperformers'])} posts earned engagement unusually fast "
            "for their age (true outperformers vs posts that just sat)."
        )
    if audience.strip():
        extra += (
            "\n\nAUDIENCE INSIGHTS (mined from the comments on the top posts — the "
            "community's real pain points, questions, objections, and vocabulary):\n"
            f"{audience.strip()}\n"
            "Ground the 'angles that resonate' and 'what to avoid' sections in these real needs."
        )
    if contrast and (contrast.get("loser_traits") or contrast.get("winner_traits")):
        lt = "; ".join(contrast.get("loser_traits", [])) or "none clear"
        wt = "; ".join(contrast.get("winner_traits", [])) or "none clear"
        extra += (
            "\n\nWINNERS VS UNDER-PERFORMERS (contrast vs a lower-engagement sample; "
            "directional):\n"
            f"- Traits more common in under-performers: {lt}\n"
            f"- Traits more common in winners: {wt}\n"
            "Use this to sharpen the 'what to avoid' section with real contrasts."
        )

    prompt = f"""Analyze these top-performing posts from r/{subreddit_name} and write a concise playbook for what makes content succeed here.

{posts_text}{extra}

Cover:
1. Title structure and hooks that work
2. Tone and voice of the community
3. Post format (length, bullets vs prose, TL;DR, etc.)
4. Emoji and formatting conventions — look at the actual posts: do the top performers use emojis? If so, where (titles vs body) and how heavily? If high performers clearly avoid them, say that instead. Only call emojis a winning pattern if the data above actually shows it.
5. Attached media — each post above is tagged with "attached media" (video / image / image gallery / none). Do the highest performers attach a video or image, or are they mostly text? Call out any clear pattern (e.g. "the top post is the only one with a demo video"). Only claim media helps if the data supports it.
6. What angles and topics resonate most
7. What to avoid"""
    if timing and timing.get("usable"):
        prompt += "\n8. Best time to post (from the timing signal above, stated as a rough hint)"
    if audience.strip():
        prompt += "\n9. Unmet needs to address (from the audience insights above)"
    prompt += "\n\nBe specific. Reference actual examples from the posts above."

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def generate_post(playbook, subreddit_name, pitch, num_variants=NUM_VARIANTS,
                  style="Casual", emojis="None", guidelines="", audience="", drivers=""):
    style_instruction = STYLE_LEVELS.get(style, STYLE_LEVELS["Casual"])
    emoji_instruction = EMOJI_LEVELS.get(emojis, EMOJI_LEVELS["None"])
    edge_block = ""
    if audience.strip():
        edge_block += f"""

AUDIENCE INSIGHTS (mined from real comments — address these unmet needs and use the
community's own vocabulary so the post feels native, not pitched):
{audience.strip()}"""
    if drivers.strip():
        edge_block += f"""

MEASURED TITLE DRIVERS (directional signal from this subreddit's top posts — bias the
titles toward these where it stays natural, never force them):
{drivers.strip()}"""
    guidelines_block = ""
    if guidelines.strip():
        guidelines_block = f"""

SUBREDDIT GUIDELINES (from the sidebar — do not violate them):
{guidelines.strip()}

Every variant must comply with these guidelines. If the user's pitch fundamentally
conflicts with a guideline (for example the sub bans self-promotion), still write the
post as natively as possible, and use the NOTE field to warn the user about the specific
risk and suggest how to soften it (e.g. share value first, mention the product only if asked)."""
    prompt = f"""Write {num_variants} distinct high-performing Reddit post variants for r/{subreddit_name} based on this playbook and the user's pitch.

PLAYBOOK:
{playbook}

USER'S PITCH:
{pitch}{edge_block}{guidelines_block}

CRITICAL WEIGHTING — THE TITLE IS ~60% OF SUCCESS:
Reddit users decide whether to click and engage almost entirely on the title/hook. Treat the title as roughly 60% of what determines whether a post succeeds, and invest the majority of your craft there. A mediocre body under a great title beats a great body under a weak title.

SOUND LIKE A REAL PERSON, NOT AI — THIS IS NON-NEGOTIABLE:
The biggest failure mode is sounding "written." Real Reddit posts are short, a little flat, and slightly sloppy. Follow these rules:
- Keep bodies SHORT: aim for 60-110 words. Shorter is more believable than longer. Cut anything that isn't load-bearing.
- Do NOT put a punchline or emotional beat on every paragraph. Most lines should be plain and unremarkable. Let the post be a little boring in places — real ones are.
- BAN these AI tells: em-dashes (use periods or commas), the "Not X. Not Y. Just Z." parallel construction, rule-of-three lists, and a neat rhetorical-question or mic-drop closer (e.g. "I am not a complicated person.").
- WRITING STYLE (match this level exactly): {style_instruction}
- EMOJIS (follow exactly): {emoji_instruction}
- Underwrite the emotion. State things plainly instead of dramatizing them. Trust the reader.

Process:
1. First, brainstorm 6-8 candidate titles, each using one of the PROVEN hook patterns from the playbook above (e.g. specific-number hook, confession/admission, comparison/contrast, lead-with-the-human-situation). Make each concrete and native to r/{subreddit_name} — never generic enough to belong in another subreddit. Keep titles understated, not punchy or overwritten.
2. Select the {num_variants} strongest and MOST DISTINCT titles — each variant must use a DIFFERENT hook strategy so the user can A/B test angles.
3. For each chosen title, write a SHORT body (60-110 words) that delivers on its promise, follows the playbook's tone, and feels genuinely helpful — not a transparent ad. Promotion should come through authenticity and value, not hype.

Format your response EXACTLY like this, with no extra commentary before or after:

=== VARIANT 1 ===
HOOK: <name the hook pattern this title uses>
TITLE: <the title>
BODY:
<the body>
NOTE: <a rule-compliance caution if anything risks removal, otherwise "none">

=== VARIANT 2 ===
HOOK: ...
TITLE: ...
BODY:
...
NOTE: ...

(continue through VARIANT {num_variants})"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def md_safe(text):
    """Escape '$' so Streamlit's markdown doesn't render dollar amounts as LaTeX math."""
    return (text or "").replace("$", "\\$")


def parse_variants(text):
    """Split the model output into [{hook, title, body}] using the VARIANT markers."""
    variants = []
    for chunk in re.split(r"===\s*VARIANT\s*\d+\s*===", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        hook = title = note = ""
        body_lines = []
        in_body = False
        for line in chunk.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("HOOK:"):
                hook = stripped[5:].strip()
            elif upper.startswith("TITLE:"):
                title = stripped[6:].strip()
            elif upper.startswith("BODY:"):
                in_body = True
                rest = stripped[5:].strip()
                if rest:
                    body_lines.append(rest)
            elif upper.startswith("NOTE:"):
                note = stripped[5:].strip()
                in_body = False  # NOTE ends the body
            elif in_body:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if title or body:
            variants.append({"hook": hook, "title": title, "body": body, "note": note})
    return variants


# ---------------------------------------------------------------------------
# Auth (Supabase email/password) + billing (Stripe) gate
# ---------------------------------------------------------------------------
# Requires a Supabase table `users`:
#   create table users (
#     email text primary key,
#     subscription_status text,
#     run_month text,
#     runs_this_month int default 0
#   );
FREE_RUN_LIMIT = 3


def _current_month():
    return datetime.now(timezone.utc).strftime("%Y-%m")


def upsert_user(email, subscription_status):
    sb = get_supabase()
    if not sb:
        return
    try:
        sb.table("users").upsert(
            {"email": email, "subscription_status": subscription_status},
            on_conflict="email",
        ).execute()
    except Exception:
        pass


def get_user_row(email):
    sb = get_supabase()
    if not sb:
        return None
    try:
        res = sb.table("users").select("*").eq("email", email).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None


def get_run_count(email):
    """Runs used in the current month (0 if the stored month is stale)."""
    row = get_user_row(email)
    if not row or row.get("run_month") != _current_month():
        return 0
    return int(row.get("runs_this_month") or 0)


def increment_run(email):
    sb = get_supabase()
    if not sb:
        return
    month = _current_month()
    row = get_user_row(email)
    if row and row.get("run_month") == month:
        new_count = int(row.get("runs_this_month") or 0) + 1
    else:
        new_count = 1
    try:
        sb.table("users").upsert(
            {"email": email, "run_month": month, "runs_this_month": new_count},
            on_conflict="email",
        ).execute()
    except Exception:
        pass


def has_active_subscription(email):
    """True if Stripe shows an active subscription for this email."""
    if not STRIPE_SECRET_KEY:
        return False
    try:
        customers = stripe.Customer.list(email=email, limit=1)
        if not customers.data:
            return False
        subs = stripe.Subscription.list(
            customer=customers.data[0].id, status="active", limit=1
        )
        return len(subs.data) > 0
    except Exception:
        return False


def _post_login(email):
    active = has_active_subscription(email)
    upsert_user(email, "active" if active else "inactive")
    st.session_state.user_email = email
    st.session_state.subscription_active = active
    st.rerun()


def _sign_out():
    sb = get_supabase()
    if sb:
        try:
            sb.auth.sign_out()
        except Exception:
            pass
    for k in ("user_email", "subscription_active"):
        st.session_state.pop(k, None)
    st.rerun()


def render_auth():
    """Login/signup form — the ONLY thing shown when logged out."""
    st.title("Reddit Post Analyzer")
    sb = get_supabase()
    if not sb:
        st.error("Auth is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.")
        return
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        email = st.text_input("Email", key="login_email")
        pw = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log in", type="primary"):
            try:
                sb.auth.sign_in_with_password({"email": email, "password": pw})
                _post_login(email)
            except Exception as e:
                st.error(f"Login failed: {e}")
    with tab_signup:
        email2 = st.text_input("Email", key="signup_email")
        pw2 = st.text_input("Password", type="password", key="signup_pw")
        if st.button("Create account"):
            try:
                sb.auth.sign_up({"email": email2, "password": pw2})
                st.success(
                    "Account created. Log in on the other tab "
                    "(confirm your email first if your project requires it)."
                )
            except Exception as e:
                st.error(f"Sign up failed: {e}")


def render_paywall(email):
    """Upgrade page — the ONLY thing shown when out of free runs and not subscribed."""
    st.title("Upgrade to Pro")
    st.write(
        f"You've used your {FREE_RUN_LIMIT} free runs this month. "
        "Upgrade to Pro for $15/month for unlimited runs."
    )
    if STRIPE_PAYMENT_LINK:
        st.link_button("Upgrade — $15/month", STRIPE_PAYMENT_LINK, type="primary")
    else:
        st.error("Billing is not configured. Set STRIPE_PAYMENT_LINK.")
    st.caption(f"Signed in as {email}")
    if st.button("Log out"):
        _sign_out()


# UI
st.set_page_config(page_title="Reddit Post Analyzer", layout="centered")

# Auth gate: when logged out, show only the login/signup form.
if not st.session_state.get("user_email"):
    render_auth()
    st.stop()

USER_EMAIL = st.session_state.user_email

# Billing gate: free tier is FREE_RUN_LIMIT runs/month unless subscribed.
if not st.session_state.get("subscription_active") and get_run_count(USER_EMAIL) >= FREE_RUN_LIMIT:
    render_paywall(USER_EMAIL)
    st.stop()

st.title("Reddit Post Analyzer")

with st.sidebar:
    st.caption(f"Signed in as {USER_EMAIL}")
    if st.session_state.get("subscription_active"):
        st.caption("Plan: Pro (unlimited)")
    else:
        _used = get_run_count(USER_EMAIL)
        st.caption(
            f"Plan: Free · {max(FREE_RUN_LIMIT - _used, 0)} of {FREE_RUN_LIMIT} runs left this month"
        )
    if st.button("Log out"):
        _sign_out()

if "playbook" not in st.session_state:
    st.session_state.playbook = None
if "subreddit" not in st.session_state:
    st.session_state.subreddit = ""
if "posts" not in st.session_state:
    st.session_state.posts = []
if "reco" not in st.session_state:
    st.session_state.reco = {"style": "Casual", "emojis": "None"}
if "guidelines" not in st.session_state:
    st.session_state.guidelines = ""
if "timing" not in st.session_state:
    st.session_state.timing = None
if "patterns" not in st.session_state:
    st.session_state.patterns = None
if "audience" not in st.session_state:
    st.session_state.audience = ""
if "contrast" not in st.session_state:
    st.session_state.contrast = None

st.header("Step 1: Analyze a Subreddit")

col1, col2 = st.columns([2, 1])
with col1:
    subreddit_input = st.text_input("Subreddit (without r/)", placeholder="entrepreneur")
with col2:
    time_window = st.selectbox(
        "Time range",
        list(TIME_WINDOWS.keys()),
        index=2,  # default to "Past month"
    )

with st.expander("⚙️ Deeper analysis (optional, costs more)"):
    mine_comments = st.checkbox(
        "Mine top comments for audience pain points & vocabulary",
        help="Runs an extra scrape + an extra AI pass. Slower and uses more credits. "
             "Biggest source of new signal — what the community actually struggles with.",
    )
    contrast_losers = st.checkbox(
        "Compare winners vs under-performers",
        help="Runs one extra scrape (no extra AI cost) to find what flops here, making "
             "'what to avoid' data-driven instead of guessed.",
    )

if st.button("Analyze", type="primary"):
    if subreddit_input:
        try:
            with st.spinner(f"Fetching top posts from r/{subreddit_input}..."):
                posts = fetch_posts(subreddit_input, time_window)
                guidelines = fetch_subreddit_info(subreddit_input)
            # FREE edge analyses — always run.
            timing = analyze_timing(posts)
            patterns = analyze_patterns(posts)
            # Opt-in costly analyses.
            audience = ""
            if mine_comments:
                with st.spinner("Mining comments for audience insights..."):
                    comments = fetch_comments(posts)
                    audience = analyze_audience(comments, subreddit_input)
            contrast = None
            if contrast_losers:
                with st.spinner("Finding under-performers to contrast..."):
                    losers = fetch_losers(subreddit_input, time_window)
                    contrast = analyze_contrast(posts, losers)
            with st.spinner("Generating playbook..."):
                playbook = generate_playbook(
                    posts, subreddit_input,
                    timing=timing, patterns=patterns,
                    audience=audience, contrast=contrast,
                )
            st.session_state.playbook = playbook
            st.session_state.subreddit = subreddit_input
            st.session_state.posts = posts
            st.session_state.reco = recommend_settings(posts)
            st.session_state.guidelines = guidelines
            st.session_state.timing = timing
            st.session_state.patterns = patterns
            st.session_state.audience = audience
            st.session_state.contrast = contrast
            increment_run(USER_EMAIL)
        except RuntimeError as e:
            st.error(str(e))  # our own clear, actionable messages
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.warning("Enter a subreddit name first.")

if st.session_state.playbook:
    st.subheader(f"r/{st.session_state.subreddit} Playbook")

    if st.session_state.posts:
        with st.expander(f"📋 The {len(st.session_state.posts)} ranked posts behind this playbook"):
            st.caption("Ranked by engagement (upvotes + comments). Open them to read the real posts.")
            for i, p in enumerate(st.session_state.posts, 1):
                media = f" · {p['media']}" if p.get("media") and p["media"] != "none" else ""
                title = md_safe(p["title"])
                link = f"[{title}]({p['url']})" if p.get("url") else title
                st.markdown(
                    f"{i}. {link}  \n"
                    f"   <small>{p['score']} pts · {p['comments']} comments · "
                    f"{p['engagement']} engagement{media}</small>",
                    unsafe_allow_html=True,
                )

    patterns = st.session_state.patterns
    if patterns and patterns.get("boolean_drivers"):
        with st.expander("📊 Measured title drivers"):
            st.caption(
                f"Computed across {patterns['n']} posts — directional signal, not proof "
                "(small sample). Each shows avg engagement with vs without the feature."
            )
            for d in patterns["boolean_drivers"]:
                sign = "+" if d["lift_pct"] >= 0 else ""
                st.markdown(
                    f"- **{d['feature']}**: {sign}{d['lift_pct']:.0f}% engagement  \n"
                    f"   <small>{d['with_mean']:.0f} avg with (n={d['n_with']}) vs "
                    f"{d['without_mean']:.0f} without (n={d['n_without']})</small>",
                    unsafe_allow_html=True,
                )
            wc = patterns.get("word_count")
            if wc:
                better = "Shorter" if wc["short_mean"] > wc["long_mean"] else "Longer"
                st.markdown(
                    f"- **title length**: {better} titles do better "
                    f"(median split at {wc['split']:.0f} words)"
                )

    timing = st.session_state.timing
    if timing:
        with st.expander("⏱️ Timing & velocity"):
            if not timing.get("usable"):
                st.caption("Timestamps weren't available for enough posts to judge timing.")
            else:
                st.caption(
                    "By engagement velocity (score ÷ age). Directional only — Reddit votes "
                    "plateau, so treat this as a rough hint, not a guaranteed best time (UTC)."
                )
                days = ", ".join(f"**{d}**" for d, _, _ in timing["best_dow"])
                hrs = ", ".join(f"**{h:02d}:00**" for h, _, _ in timing["best_hours"])
                st.markdown(f"- Highest-velocity days: {days}")
                st.markdown(f"- Highest-velocity hours (UTC): {hrs}")
                st.markdown(
                    f"- **{len(timing['outperformers'])}** posts earned engagement unusually "
                    "fast for their age (true outperformers, not posts that just sat)."
                )

    if st.session_state.audience:
        with st.expander("👥 Audience insights (from comments)", expanded=True):
            st.caption("Mined from top comments on the top posts — real pain points and vocabulary.")
            st.markdown(md_safe(st.session_state.audience))

    contrast = st.session_state.contrast
    if contrast and (contrast.get("loser_traits") or contrast.get("winner_traits")):
        with st.expander("🆚 Winners vs under-performers"):
            st.caption("Title traits that over-index in each group (directional, ≥20pp gap).")
            if contrast.get("winner_traits"):
                st.markdown("**More common in winners:**")
                for t in contrast["winner_traits"]:
                    st.markdown(f"- {t}")
            if contrast.get("loser_traits"):
                st.markdown("**More common in under-performers (avoid):**")
                for t in contrast["loser_traits"]:
                    st.markdown(f"- {t}")

    st.markdown(md_safe(st.session_state.playbook))

    st.divider()
    st.header("Step 2: Generate Your Post")

    st.caption(
        "The more you give the model, the better it can match the playbook. Include the real "
        "story: what it is, who it's for, why you built it, any numbers or results, and a "
        "personal angle or struggle. Write it the way you'd tell a friend."
    )
    pitch = st.text_area(
        "What are you promoting? (more detail = better posts)",
        height=200,
        placeholder=(
            "e.g. I built a Chrome extension that blocks distracting sites during work hours. "
            "It's free. I made it after failing my finals because I couldn't stop opening "
            "YouTube. Took 3 months, ~40 users so far. I want to share it without sounding "
            "salesy, and I'm a bit nervous about posting."
        ),
    )

    if st.session_state.guidelines:
        with st.expander("📜 Subreddit guidelines"):
            st.caption(
                "This is the public sidebar description, not the full numbered rules list. "
                "Always check the subreddit's posted rules before submitting. Generated posts "
                "try to follow these, and flag anything risky."
            )
            st.markdown(md_safe(st.session_state.guidelines))

    reco = st.session_state.reco
    st.caption(
        f"💡 Recommended from r/{st.session_state.subreddit}'s top posts: "
        f"**{reco['style']}** style, **{reco['emojis']}** emojis. The sliders default to this — adjust freely."
    )
    opt1, opt2 = st.columns(2)
    with opt1:
        style = st.select_slider(
            "Writing style",
            options=list(STYLE_LEVELS.keys()),
            value=reco["style"],
            help="How polished vs. off-the-cuff the posts read. Defaulted to match this subreddit.",
        )
    with opt2:
        emojis = st.select_slider(
            "Emojis",
            options=list(EMOJI_LEVELS.keys()),
            value=reco["emojis"],
            help="Defaulted to match how often this subreddit's top posts use emojis.",
        )

    if st.button("Generate Post", type="primary"):
        if pitch:
            with st.spinner(f"Writing {NUM_VARIANTS} variants ({style.lower()}, title-weighted)..."):
                post_draft = generate_post(
                    st.session_state.playbook,
                    st.session_state.subreddit,
                    pitch,
                    style=style,
                    emojis=emojis,
                    guidelines=st.session_state.guidelines,
                    audience=st.session_state.audience,
                    drivers=format_drivers(st.session_state.patterns),
                )
            variants = parse_variants(post_draft)
            if variants:
                st.subheader("Your Post Variants")
                st.caption("Each variant leads with a different proven hook — the title carries ~60% of the weight.")
                tabs = st.tabs([f"Variant {i + 1}" for i in range(len(variants))])
                for tab, v in zip(tabs, variants):
                    with tab:
                        if v["hook"]:
                            st.caption(f"Hook pattern: {v['hook']}")
                        if v["title"]:
                            st.markdown(f"### {md_safe(v['title'])}")
                        st.markdown(md_safe(v["body"]))
                        note = v.get("note", "")
                        if note and note.lower() not in ("none", "n/a", ""):
                            st.warning(f"⚠️ Compliance: {note}")
                        st.code(
                            f"{v['title']}\n\n{v['body']}",
                            language=None,
                        )  # copy-ready plain text
                st.info(
                    "✏️ Before you post: pick your favorite, then change a few words and tweak "
                    "the punctuation by hand. Add a small personal detail only you would know. "
                    "Those tiny edits are what make it read as genuinely yours, not AI-written."
                )
            else:
                # Fallback: model didn't follow the variant format; show raw output.
                st.subheader("Your Post Draft")
                st.markdown(md_safe(post_draft))
        else:
            st.warning("Add your pitch first.")

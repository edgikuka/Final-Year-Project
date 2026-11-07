from __future__ import annotations

import random
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.config.config import load_config
from src.driver.webdriver_setup import setup_web_driver

# =========================
# CONFIG
# =========================
INPUT_CSV = Path("final POIs with usernames.csv")
USERNAME_COL = "x username"


# =========================
# HELPERS
# =========================
def norm_username(x: str) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    return s.lstrip("@").strip() or None

def human_pause(min_s=2.6, max_s=5.0):
    time.sleep(random.uniform(min_s,max_s))

def human_scroll(driver,max_scrolls=3):
    for _ in range(random.randint(1,max_scrolls)):
        driver.execute_script(
            "window.scrollBy(0, arguments[0]);",
            random.randint(200,600)
        )
        time.sleep(random.uniform(0.8,1.6))

def parse_compact_number(s: str | None) -> int | None:
    if not s:
        return None
    s = s.replace(",", "").strip()
    m = re.match(r"^(\d+(?:\.\d+)?)([KMB])?$", s, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    suf = (m.group(2) or "").upper()
    if suf == "K":
        val *= 1_000
    elif suf == "M":
        val *= 1_000_000
    elif suf == "B":
        val *= 1_000_000_000
    return int(val)


def extract_first_number(text: str | None) -> int | None:
    if not text:
        return None
    t = text.replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)([KMB])?", t, re.IGNORECASE)
    if not m:
        return None
    return parse_compact_number(m.group(0))


def safe_text(driver, css: str) -> str | None:
    try:
        v = driver.find_element(By.CSS_SELECTOR, css).text
        v = v.strip() if v else ""
        return v or None
    except Exception:
        return None


def safe_attr(driver, css: str, attr: str) -> str | None:
    try:
        v = driver.find_element(By.CSS_SELECTOR, css).get_attribute(attr)
        v = v.strip() if v else ""
        return v or None
    except Exception:
        return None


def wait_for_profile(driver, timeout: int = 25) -> None:
    wait = WebDriverWait(driver, timeout)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[data-testid="primaryColumn"]')))


def wait_for_counts_in_primary_column(driver, timeout: int = 25) -> None:
    """
    Wait until the primary column text contains BOTH counts (following + followers).
    """
    def ready(drv) -> bool:
        try:
            col = drv.find_element(By.CSS_SELECTOR, 'div[data-testid="primaryColumn"]')
            txt = (col.get_attribute("innerText") or "").strip()
            if not txt:
                return False
            low = txt.lower()

            # Must contain "Following" and either "Followers" or Hebrew followers "עוקב"
            has_following = "following" in low and re.search(r"\d", txt) is not None
            has_followers = ("followers" in low or "עוקב" in txt) and re.search(r"\d", txt) is not None

            return has_following and has_followers
        except Exception:
            return False

    WebDriverWait(driver, timeout).until(lambda d: ready(d))


def open_profile_with_retry(driver, username: str, tries: int = 3) -> None:
    url = f"https://x.com/{username}"
    last_err = None
    for attempt in range(1, tries + 1):
        driver.get(url)
        try:
            wait_for_profile(driver, timeout=25)
            wait_for_counts_in_primary_column(driver, timeout=25)
            return
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to load profile @{username}: {last_err}")


def parse_follow_counts_from_text(text: str) -> tuple[int | None, int | None]:
    """
    Extract (following, followers) from a big text blob.
    Looks for lines like:
      '704 Following'
      '2,218 Followers'
    Also supports Hebrew followers: 'עוקבים'
    """
    following = None
    followers = None

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines:
        low = ln.lower()

        if "following" in low:
            n = extract_first_number(ln)
            if n is not None:
                following = n

        if "followers" in low or "עוקב" in ln:
            n = extract_first_number(ln)
            if n is not None:
                followers = n

    return following, followers


def get_follow_counts(driver) -> tuple[int | None, int | None]:
    """
    FIX: Scan the entire primaryColumn (which definitely includes the counts in your UI).
    """
    try:
        col = driver.find_element(By.CSS_SELECTOR, 'div[data-testid="primaryColumn"]')
        txt = (col.get_attribute("innerText") or "").strip()
        return parse_follow_counts_from_text(txt)
    except Exception:
        return None, None


def get_post_count(driver) -> int | None:
    keywords = ["posts", "tweets", "פוסטים"]

    try:
        tab_links = driver.find_elements(By.CSS_SELECTOR, 'div[role="tablist"] a')
        for a in tab_links:
            txt = (a.get_attribute("innerText") or a.text or "").strip()
            if not txt:
                continue
            if any(k in txt.lower() for k in keywords) or re.search(r"\d", txt):
                n = extract_first_number(txt)
                if n is not None:
                    return n
    except Exception:
        pass

    # fallback: scan primary column
    try:
        col = driver.find_element(By.CSS_SELECTOR, 'div[data-testid="primaryColumn"]')
        txt = (col.get_attribute("innerText") or "").strip()
        for key in ["posts", "tweets", "פוסטים"]:
            idx = txt.lower().find(key)
            if idx != -1:
                window = txt[max(0, idx - 40): idx + 40]
                n = extract_first_number(window)
                if n is not None:
                    return n
    except Exception:
        pass

    return None


def scrape_user(driver, username: str) -> dict:
    open_profile_with_retry(driver, username, tries=3)

    human_pause(3.0,6.0)
    human_scroll(driver)
    human_pause(2.0,4.0)

    url = f"https://x.com/{username}"

    name = safe_text(driver, 'div[data-testid="UserName"] span')
    bio = safe_text(driver, 'div[data-testid="UserDescription"]')

    # location/joined are usually inside UserProfileHeader_Items if present
    location = None
    joined_text = None
    try:
        items = driver.find_element(By.CSS_SELECTOR, 'div[data-testid="UserProfileHeader_Items"]')
        items_text = (items.get_attribute("innerText") or items.text or "").strip()
        lines = [ln.strip() for ln in items_text.split("\n") if ln.strip()]
        for ln in lines:
            low = ln.lower()
            if low.startswith("joined") or "joined" in low or "הצטרף" in low:
                joined_text = ln
            else:
                if location is None:
                    location = ln
    except Exception:
        pass

    try:
        driver.find_element(By.CSS_SELECTOR, '[data-testid="icon-verified"]')
        verified = True
    except Exception:
        verified = False

    following_count, followers_count = get_follow_counts(driver)
    tweet_count = get_post_count(driver)

    profile_image_url = safe_attr(driver, 'img[src*="profile_images"]', "src")
    banner_image_url = safe_attr(driver, 'img[src*="profile_banners"]', "src")

    return {
        "x_username": username,
        "name": name,
        "bio": bio,
        "location": location,
        "joined_text": joined_text,
        "followers_count": followers_count,
        "following_count": following_count,
        "tweet_count": tweet_count,
        "profile_image_url": profile_image_url,
        "banner_image_url": banner_image_url,
        "verified": verified,
        "profile_url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# =========================
# MAIN
# =========================
def main():
    cfg = load_config(Path("config/config.yaml"))
    driver = setup_web_driver(debug_address=cfg.chrome_debug_address)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "users_metadata.csv"

    df = pd.read_csv(INPUT_CSV)
    usernames = (
        df[USERNAME_COL]
        .map(norm_username)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    done = set()
    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv, dtype=str)
            if "x_username" in prev.columns:
                done = set(prev["x_username"].dropna().astype(str).tolist())
        except Exception:
            pass

    try:
        total = len(usernames)
        for i, u in enumerate(usernames, 1):
            if u in done:
                continue

            print(f"[{i}/{total}] scraping @{u}")

            try:
                meta = scrape_user(driver, u)
            except Exception:
                meta = {
                    "x_username": u,
                    "name": None,
                    "bio": None,
                    "location": None,
                    "joined_text": None,
                    "followers_count": None,
                    "following_count": None,
                    "tweet_count": None,
                    "profile_image_url": None,
                    "banner_image_url": None,
                    "verified": None,
                    "profile_url": f"https://x.com/{u}",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }

            pd.DataFrame([meta]).to_csv(
                out_csv,
                mode="a",
                header=not out_csv.exists(),
                index=False,
                encoding="utf-8-sig",
            )

            time.sleep(1.5)

        print(f"Done. Wrote metadata to: {out_csv}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
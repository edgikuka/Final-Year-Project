from pathlib import Path
import pandas as pd

# =========================
# FILES
# =========================
ORIGINAL_CSV = Path("final POIs with usernames.csv")
META_CSV = Path("data/users_metadata.csv")
OUT_CSV = Path("final POIs with usernames + metadata.csv")

USERNAME_COL = "x username"   # column in original CSV


def norm_username(x):
    """
    Normalize usernames so they match:
    - remove @
    - strip spaces
    - handle NaN
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    return s.lstrip("@").strip()


def main():
    # Load files
    df = pd.read_csv(ORIGINAL_CSV)
    meta = pd.read_csv(META_CSV)

    # Normalize join keys
    df["_u"] = df[USERNAME_COL].map(norm_username)
    meta["_u"] = meta["x_username"].map(norm_username)

    # Drop duplicate usernames in metadata (keep first)
    meta = meta.drop_duplicates(subset="_u")

    # Remove x_username column to avoid duplication
    meta = meta.drop(columns=["x_username"], errors="ignore")

    # Merge (LEFT JOIN = keep all original rows)
    merged = df.merge(meta, on="_u", how="left")

    # Cleanup
    merged = merged.drop(columns=["_u"])

    # Save
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"✅ Merged file written to: {OUT_CSV}")
    print(f"Rows: {len(merged)} | Metadata rows: {len(meta)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Review Radar: where our paper sits among ICLR submissions, from the public OpenReview score mirror.

    python3 scripts/review_radar.py                      # uses cached score lists in ~/.cache/tsfm_radar (downloads if missing)
    python3 scripts/review_radar.py --refresh            # re-download the lists (Paper Copilot's paperlists mirror of OpenReview)
    python3 scripts/review_radar.py --abstract paper/main.tex --out docs/review_radar.html

What it computes
  1. Neighbours: TF-IDF cosine similarity between our abstract (+ keywords) and every decided ICLR 2025/2026 paper's
     title + abstract + keywords; the top-N with their decisions, per-reviewer ratings and sub-scores.
  2. Empirical acceptance curve: P(accept | mean rating) over all decided papers, and within our primary area.
  3. Score predictors for our abstract: (a) similarity-weighted kNN over the neighbours' mean ratings; (b) a ridge
     regression on TF-IDF features fitted to mean rating (cross-validated MAE reported); (c) a logistic model of
     acceptance. These are priors from text alone; they cannot see the evidence in the paper, which is the point.
  4. Sub-score profile (soundness / presentation / contribution) of accepted vs rejected neighbours: the shape a
     paper needs to have.
Everything is written into one self-contained HTML page (embedded JSON, no external requests) plus a JSON summary.
"""
from __future__ import annotations

import argparse, html, json, os, re, sys, urllib.request
import numpy as np

CACHE = os.path.expanduser("~/.cache/tsfm_radar")
SRC = {"iclr2025": "https://media.githubusercontent.com/media/papercopilot/paperlists/main/iclr/iclr2025.json",
       "iclr2026": "https://media.githubusercontent.com/media/papercopilot/paperlists/main/iclr/iclr2026.json"}
ACCEPT = ("Poster", "Oral", "Spotlight", "ICLR 2026 ConditionalPoster", "ICLR 2026 ConditionalOral")
DECIDED = ACCEPT + ("Reject",)


def load_lists(refresh=False):
    os.makedirs(CACHE, exist_ok=True); out = []
    for name, url in SRC.items():
        p = os.path.join(CACHE, name + ".json")
        if refresh or not os.path.exists(p):
            print("downloading", url, flush=True); urllib.request.urlretrieve(url, p)
        for e in json.load(open(p)):
            if e.get("status") not in DECIDED or not e.get("rating"): continue
            try: r = [float(x) for x in e["rating"].split(";")]
            except ValueError: continue
            if min(r) < 1 or len(r) < 2: continue                                   # malformed entries (ratings are on a 1-10 scale)
            def sub(k):
                try: return [float(x) for x in e.get(k, "").split(";")] if e.get(k) else []
                except ValueError: return []
            out.append({"year": name[-4:], "id": e.get("openreview_forum") or e.get("id"), "title": e["title"], "abstract": e.get("abstract", ""), "keywords": e.get("keywords", ""),
                        "area": e.get("primary_area", ""), "status": e["status"], "accept": e["status"] in ACCEPT, "ratings": r, "avg": float(np.mean(r)),
                        "soundness": sub("soundness"), "presentation": sub("presentation"), "contribution": sub("contribution"), "confidence": sub("confidence")})
    return out


def our_text(tex_path):
    s = open(tex_path).read(); m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S); a = m.group(1) if m else ""
    a = re.sub(r"\\todo\{[^}]*\}", "", a); a = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", a); a = re.sub(r"[{}$]", " ", a)
    t = re.search(r"\\title\{(.*?)\}", s, re.S); title = re.sub(r"\\\\", " ", t.group(1)) if t else ""
    return title.strip(), re.sub(r"\s+", " ", a).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--abstract", default="paper/main.tex"); ap.add_argument("--keywords", default="time series foundation model; reinforcement learning post-training; calibration; proper scoring rules; CRPS; GRPO; forecasting")
    ap.add_argument("--area", default="learning on time series and dynamical systems"); ap.add_argument("--top", type=int, default=40); ap.add_argument("--out", default="docs/review_radar.html"); ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import cross_val_predict
    papers = load_lists(a.refresh); title, abstract = our_text(a.abstract); ours = f"{title}. {abstract} {a.keywords}"
    docs = [f"{p['title']}. {p['abstract']} {p['keywords']}" for p in papers]
    vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english"); X = vec.fit_transform(docs); x = vec.transform([ours])
    sim = (X @ x.T).toarray().ravel(); order = np.argsort(-sim)[:a.top]
    neigh = [{**{k: v for k, v in papers[i].items() if k != "abstract"}, "sim": float(sim[i]), "abstract": papers[i]["abstract"][:600]} for i in order]
    # empirical acceptance curve
    avgs = np.array([p["avg"] for p in papers]); acc = np.array([p["accept"] for p in papers]); yrs = np.array([p["year"] for p in papers]); areas = np.array([p["area"] for p in papers])
    def curve(mask):
        pts = []
        for lo in np.arange(2.0, 8.01, 0.5):
            m = mask & (np.abs(avgs - lo) < 0.25)
            if m.sum() >= 20: pts.append({"avg": float(lo), "p": float(acc[m].mean()), "n": int(m.sum())})
        return pts
    curve_all = curve(yrs == "2026"); curve_area = curve((yrs == "2026") & (areas == a.area)); curve_2025 = curve(yrs == "2025")
    # predictors
    w = np.array([n["sim"] for n in neigh]); w = w / w.sum(); knn_rating = float(sum(wi * n["avg"] for wi, n in zip(w, neigh))); knn_accept = float(sum(wi * n["accept"] for wi, n in zip(w, neigh)))
    m26 = yrs == "2026"; ridge = Ridge(alpha=3.0); cv_pred = cross_val_predict(ridge, X[m26], avgs[m26], cv=5); cv_mae = float(np.mean(np.abs(cv_pred - avgs[m26])))
    ridge.fit(X[m26], avgs[m26]); ridge_rating = float(ridge.predict(x)[0])
    logit = LogisticRegression(max_iter=2000, C=2.0); logit.fit(X[m26], acc[m26]); p_text = float(logit.predict_proba(x)[0, 1])
    from sklearn.metrics import roc_auc_score; auc = float(roc_auc_score(acc[m26], cross_val_predict(LogisticRegression(max_iter=2000, C=2.0), X[m26], acc[m26], cv=5, method="predict_proba")[:, 1]))
    # sub-score profiles among neighbours
    def prof(sel):
        rows = [n for n in neigh if n["accept"] == sel and n["soundness"]]
        if not rows: return None
        return {k: float(np.mean([np.mean(n[k]) for n in rows])) for k in ("soundness", "presentation", "contribution")} | {"n": len(rows), "avg": float(np.mean([n["avg"] for n in rows]))}
    area_stats = {"n": int(((yrs == "2026") & (areas == a.area)).sum()), "accept_rate": float(acc[(yrs == "2026") & (areas == a.area)].mean()) if ((yrs == "2026") & (areas == a.area)).any() else None}
    summary = {"title": title, "abstract": abstract, "n_corpus": len(papers), "area": a.area, "area_stats": area_stats,
               "curve_all_2026": curve_all, "curve_area_2026": curve_area, "curve_2025": curve_2025,
               "predictions": {"knn_rating": knn_rating, "knn_accept": knn_accept, "ridge_rating": ridge_rating, "ridge_cv_mae": cv_mae, "text_logit_accept": p_text, "text_logit_cv_auc": auc},
               "profile_accepted": prof(True), "profile_rejected": prof(False), "neighbours": neigh,
               "overall_2026": {"n": int(m26.sum()), "accept_rate": float(acc[m26].mean()), "mean_rating_accept": float(avgs[m26 & acc].mean()), "mean_rating_reject": float(avgs[m26 & ~acc].mean())}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(summary, open(a.out.replace(".html", ".json"), "w"), indent=1)
    open(a.out, "w").write(render(summary)); print("wrote", a.out, "and", a.out.replace(".html", ".json"))
    print(json.dumps(summary["predictions"], indent=1)); print("area", area_stats)


def render(S):
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_radar_template.html")).read()
    return tpl.replace("/*__DATA__*/", "const DATA = " + json.dumps(S) + ";")


if __name__ == "__main__":
    main()

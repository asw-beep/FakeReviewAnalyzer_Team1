// =============================================================================
// Review Trust Analyzer — Frontend JavaScript
// =============================================================================

const API_URL = "http://localhost:8787";

// ─── Prediction ─────────────────────────────────────────────────────────────

async function predictReview() {
    const textarea = document.getElementById("review-input");
    const text = textarea.value.trim();
    if (!text) {
        textarea.focus();
        return;
    }

    const btn = document.getElementById("predict-btn");
    const loading = document.getElementById("loading");
    const resultCard = document.getElementById("result-card");

    btn.classList.add("hidden");
    loading.classList.remove("hidden");
    resultCard.classList.add("hidden");

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review_text: text }),
        });

        if (!res.ok) throw new Error("API error");
        const data = await res.json();
        showResult(data);
    } catch (err) {
        console.error(err);
        // Fallback: simple heuristic-based prediction when API is down
        showResult(heuristicPredict(text));
    } finally {
        btn.classList.remove("hidden");
        loading.classList.add("hidden");
    }
}

function showResult(data) {
    const resultCard = document.getElementById("result-card");
    const icon = document.getElementById("result-icon");
    const verdict = document.getElementById("result-verdict");
    const subtitle = document.getElementById("result-subtitle");
    const fakeBar = document.getElementById("fake-bar");
    const genuineBar = document.getElementById("genuine-bar");
    const fakePct = document.getElementById("fake-pct");
    const genuinePct = document.getElementById("genuine-pct");

    const isFake = data.verdict === "Fake";

    icon.className = `result-icon ${isFake ? "fake" : "genuine"}`;
    icon.textContent = isFake ? "⚠️" : "✅";

    verdict.textContent = isFake ? "Likely Fake Review" : "Appears Genuine";
    verdict.style.color = isFake ? "var(--accent-red)" : "var(--accent-emerald)";

    subtitle.textContent = isFake
        ? "This review shows patterns consistent with computer-generated content."
        : "This review appears to be an authentic, human-written review.";

    fakePct.textContent = data.fake_prob + "%";
    genuinePct.textContent = data.genuine_prob + "%";

    resultCard.classList.remove("hidden");

    // Animate bars after a short delay
    requestAnimationFrame(() => {
        setTimeout(() => {
            fakeBar.style.width = data.fake_prob + "%";
            genuineBar.style.width = data.genuine_prob + "%";
        }, 50);
    });

    // Stats
    document.getElementById("stat-words").textContent = data.review_length ?? "—";
    document.getElementById("stat-exclaim").textContent = data.exclaim_count ?? "—";
    document.getElementById("stat-caps").textContent =
        data.caps_ratio != null ? data.caps_ratio + "%" : "—";
}

// Heuristic fallback when the R API is not running
function heuristicPredict(text) {
    const words = text.split(/\s+/).length;
    const exclaims = (text.match(/!/g) || []).length;
    const caps = (text.match(/[A-Z]/g) || []).length;
    const capsRatio = Math.round((caps / Math.max(text.length, 1)) * 10000) / 100;

    // Simple heuristic scoring
    let fakeScore = 0;
    if (words < 20) fakeScore += 20;
    if (exclaims > 3) fakeScore += 15;
    if (capsRatio > 10) fakeScore += 10;
    if (/amazing|incredible|best ever|perfect|awesome/i.test(text)) fakeScore += 15;
    if (/!!|!!!/.test(text)) fakeScore += 10;
    if (words < 10 && exclaims > 1) fakeScore += 15;

    fakeScore = Math.min(fakeScore, 95);
    const genuineScore = 100 - fakeScore;

    return {
        verdict: fakeScore > 50 ? "Fake" : "Genuine",
        fake_prob: fakeScore,
        genuine_prob: genuineScore,
        review_length: words,
        exclaim_count: exclaims,
        caps_ratio: capsRatio,
    };
}

// ─── Filter Tabs ────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    const filterBtns = document.querySelectorAll(".filter-btn");
    const figureCards = document.querySelectorAll(".figure-card");

    filterBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const filter = btn.dataset.filter;

            filterBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            figureCards.forEach((card, i) => {
                const match = filter === "all" || card.dataset.category === filter;
                card.style.display = match ? "" : "none";
                if (match) {
                    card.style.animationDelay = `${i * 0.05}s`;
                    card.style.animation = "none";
                    card.offsetHeight; // trigger reflow
                    card.style.animation = "";
                }
            });
        });
    });

    // ─── Lightbox ───────────────────────────────────────────────────────────
    figureCards.forEach((card) => {
        card.addEventListener("click", () => {
            const img = card.querySelector("img");
            const caption = card.querySelector("h3").textContent;
            openLightbox(img.src, caption);
        });
    });

    // ─── Navbar highlight on scroll ─────────────────────────────────────────
    const navLinks = document.querySelectorAll(".nav-link");
    const sections = document.querySelectorAll("section");

    window.addEventListener("scroll", () => {
        let current = "";
        sections.forEach((section) => {
            const top = section.offsetTop - 100;
            if (window.scrollY >= top) current = section.id;
        });
        navLinks.forEach((link) => {
            link.classList.toggle(
                "active",
                link.getAttribute("href") === `#${current}`
            );
        });
    });

    // ─── Enter key support ──────────────────────────────────────────────────
    document.getElementById("review-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            predictReview();
        }
    });
});

// ─── Lightbox Functions ─────────────────────────────────────────────────────

function openLightbox(src, caption) {
    const lb = document.getElementById("lightbox");
    document.getElementById("lightbox-img").src = src;
    document.getElementById("lightbox-caption").textContent = caption;
    lb.classList.remove("hidden");
    document.body.style.overflow = "hidden";
}

function closeLightbox() {
    document.getElementById("lightbox").classList.add("hidden");
    document.body.style.overflow = "";
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
});

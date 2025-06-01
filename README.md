# UChicago Trading Competition (Case 1 Focus)

## Competition Overview  
The UChicago Trading Competition runs on the xChange simulated exchange, pitting algorithmic market‑making bots against both adversarial “hitter” bots and other student teams. The competition is divided into four 15‑minute rounds (each consisting of 10 “days” of 90 seconds each). At the end of each round, positions reset and P&L is marked to settlement prices (midpoints or modeled fundamentals) and converted into points via a nonlinear scoring function. Rounds grow progressively more difficult: competing market makers tighten spreads, reduce displayed depth, and react faster to news. Consistent P&L over multiple rounds is rewarded, while extreme one‐off wins or losses have diminishing impact on the final leaderboard.


## Case 1 Overview  
In Case 1, teams trade three stocks—APT (large‑cap), DLR (mid‑cap), MKJ (small‑cap)—and two ETFs: AKAV (basket of APT + DLR + MKJ) and AKIM (daily inverse of AKAV). AKAV can be created/redeemed via a swap (for a \$5 fee), whereas AKIM rebalances internally at end‑of‑day and cannot be created or redeemed.

- **Central Limit Order Books** for each tradable asset (stocks and ETFs).  
- **Risk Enforcement:**  
  - Position limits, open order limits, and maximum outstanding volume per symbol.  
  - Breaching any limit causes immediate order rejection without explanation.  

- **Round Structure:**  
  - Each round = 10 “trading days” × 90 seconds/day.  
  - Positions carry over day‑to‑day but reset at round end.  

- **News Schedule:**  
  - **APT:** Two structured earnings updates per day (at 30s and 60s).  
  - **DLR:** Five structured petition updates per day (every 15 seconds) reporting new signatures.  
  - **MKJ:** Multiple unstructured text news events; no guaranteed predictive power.  


---

## Education & Symbology  

1. **ETFs & Swaps**  
   - **ETF (AKAV):** A single share represents 1 share of APT + 1 DLR + 1 MKJ. You can swap 1 ETF for 1 each stock (and vice versa) by paying a \$5 creation/redemption fee.  
   - **Inverse ETF (AKIM):** Daily moves opposite to AKAV. Does not allow creation/redemption; it simply resets every trading day. Because percentage up/down movements are asymmetric, holding AKIM introduces “volatility drag” over time.  

2. **Asset Characteristics**  
   - **APT (Large‑Cap Stock):** Releases quarterly earnings. Fair value ≈ P/E ratio × EPS. Earnings news arrives at fixed intervals.  
   - **DLR (Mid‑Cap Binary Stock):** A petition’s success dictates final value. Use a log‑normal growth model on cumulative signatures to forecast probability of reaching 100,000 by day 10. Settlement for DLR is binary: if petition signatures ≥ 100 000 by day 10, DLR settles at \$100; otherwise it settles at \$0 (bankruptcy). 
   - **MKJ (Small‑Cap Stock):** Receives only unstructured, often low‑quality news.

---

## Our Approach (Case 1)

1. **Fair Value Computation**  
   - **APT:**  
     - On each earnings update (at 30 s, 60 s), calculate `news_price = 10 (constant {P/E}) × EPS`.  
     - Blend with mid‑market midpoint via exponential time decay:
    (apt_val.png)

     - When news fair valule ≤ 0.15, rely solely on mid‑market until next earnings arrives.  
   - **DLR:**  
     - Maintain cumulative signatures and track “news event” index (up to 50 updates total).  
     - For each petition update, compute probability of reaching 100 000 by day 10 using log‑normal forecast:  
       \[
         \mu_{\text{final}} = \ln S_t + (50 - t)\ln\alpha,\quad
         \sigma_{\text{final}} = \sqrt{50 - t}\,\sigma,\quad
         P = 1 - \Phi\Bigl(\tfrac{\ln(100\,000) - \mu_{\text{final}}}{\sigma_{\text{final}}}\Bigr).
       \]  
     - Set `fair_DLR = 100 × P`.  
   - **MKJ:**  
     - Since unstructured news was historically noise, we opted not to trade it. We did not attempt advanced NLP given time constraints.  
   - **AKAV:**  
     - Compute `synthetic_fair = fair_APT + fair_DLR + fair_MKJ`.  
     - Add a small position‑based tilt (e.g., +0.05 × DLR position) to account for inventory imbalance.  
     - If `synthetic_fair > AKAV_market + \$5 + buffer`, execute a “toAKAV” swap; if `AKAV_market > synthetic_fair + \$5 + buffer`, execute a “fromAKAV” swap.  
   - **AKIM:**  
     - Because AKIM rebalances daily and cannot be created or redeemed, we opted not to trade it; its mid‑market was too unstable for fair modeling.

2. **Quote Placement & Spread Management**  
   - For each symbol, maintain a continuous quoting loop:  
     1. **Compute fair value** as above.  
     2. **Observe best competitor bid/ask** from local order book snapshot. If competitor bid exists, set our bid = competitor_bid + \$0.01; otherwise, default to a preset maximum edge (e.g., \$2). Mirror logic on the ask side.  
     3. **Enforce Minimum Spread** of \$0.01 and a maximum edge to avoid absurd prices.  
   - **Adjust for Inventory (Fading)**  
     - Optionally: `faded_fair = fair_value - 0.02 × position` to steer fill probabilities away from harmful inventory builds. In practice, we disabled fading during live rounds because we were losing our edge.

3. **Risk & Order Cancellation Logic**  
   - **Outstanding Volume:** If total unfilled volume + 60 shares > 120 (limit), cancel least competitive orders until volume is below threshold.  
   - **Stale Orders:** If our resting bid > new fair bid or resting ask < new fair ask, cancel those orders to avoid crossing stale quotes.  
   - **Large Spikes:** When a suspiciously large competitor order appears (e.g., > 5× normal lot), temporarily widen spread to mitigate adverse selection.


---

## Takeaways/Improvements to Make
- **Trade Against Dumb Money:** Identify bots that do not react optimally to news. Capture spread revenue from them.  
- **Selective Crossing:** If fair value deviates substantially from mid, it can be justified to cross the spread and lock in a profit instead of waiting for counterparties.  
- **Position Timing:** Holding positions to settlement can yield payoff but exposes you to large end‐of‐day moves. Consider paying the spread or swap fee to unwind when uncertainty is high.  
- **ETF Market Making Nuances:**  
  - Look for fleeting mispricings between AKAV and the sum of underlyings.  
  - Always factor in creation/redemption cost; choose not to trade if the edge (synthetic − market) ≈ fee.  
  - If willing to accept higher variance, you can take only one side of the arb (e.g., buy underlying even if ETF is mildly expensive) when you believe underlying is truly mispriced.  
- **News Impact Modeling:**  
  - APT earnings: if you act before others, you can buy below fair after positive EPS and sell above fair after negative EPS.  
  - DLR petition: be willing to aggressively scale into/out of DLR when probability crosses critical thresholds (e.g., moving from 40 %→60 % chance).  
  - MKJ unstructured: unless you have a robust NLP filter, consider ignoring or quoting tight around mid.

---

## Our Focus & Future Improvements  

1. **Ultra‑Low Latency News Processing**  
   - Further reduce reaction time to structured APT and DLR updates by dedicating a separate thread/process solely for news parsing and fair‑value recomputation.  

2. **Dynamic Drift/Volatility Estimation (DLR)**  
   - Replace static \(\alpha=1.0630\), \(\sigma=0.006\) with an online Kalman filter that re‑estimates growth parameters from the last 5 petition updates. This would adapt to sudden signature surges or decays.  

3. **NLP Filtering for MKJ**  
   - Build a lightweight classifier (e.g., TF‑IDF + logistic regression) trained on historical “useful” vs. “noise” headlines to decide when to treat unstructured news as signal.  

4. **Adaptive Spread Optimization**  
   - Use a simple multi‑armed bandit or reinforcement learning framework to adjust spread width based on recent fill success and adverse selection metrics.  

5. **Order Book Depth Awareness**  
   - Incorporate depth at top‑of‑book levels for APT/MKJ to estimate market impact before placing large quotes. This would help decide whether to use a swap or hedge via underlying for AKAV.  

6. **Risk‑Adaptive Position Sizing**  
   - Instead of fixed 40‑lot increments, size orders relative to realized volatility and competitor fill rates. In high‑volatility windows, place smaller, more frequent orders.  

By focusing primarily on Case 1 and executing the strategies outlined above, we aimed to capture structured news edges, exploit ETF arbitrage, and manage inventory risk effectively. Continuous adaptation—both within rounds (reacting to competing spreads and emerging news) and between rounds (re‑tuning quoting aggressiveness)—was essential for maintaining consistent P&L across the competition’s dynamic environment.

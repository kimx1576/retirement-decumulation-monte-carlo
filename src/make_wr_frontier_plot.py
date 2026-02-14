import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "sanityB_wr_sweep_50k.csv"   # change to step7_wr_sweep_90_95.csv if you want
OUT_PNG = "wr_frontier.png"

df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(10, 6), dpi=200)
plt.plot(df["wr"] * 100, df["baseline_success"] * 100, label="Baseline", linewidth=2)
plt.plot(df["wr"] * 100, df["guardrail_success"] * 100, label="Guardrail", linewidth=2)
plt.plot(df["wr"] * 100, df["guardrail_put_success"] * 100, label="Guardrail + Put (best)", linewidth=2)

# reference lines
plt.axhline(90, color="gray", linestyle="--", linewidth=1)
plt.axhline(95, color="gray", linestyle="--", linewidth=1)
plt.text(df["wr"].min() * 100, 90.4, "90% target", color="gray")
plt.text(df["wr"].min() * 100, 95.4, "95% target", color="gray")

plt.title("Success Rate vs Withdrawal Rate (WR)")
plt.xlabel("Withdrawal Rate (%)")
plt.ylabel("Success Rate (%)")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG)
print("Saved:", OUT_PNG)
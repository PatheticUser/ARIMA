Got it, Rameez. Here's a **quick, clear comparative summary** of your ARIMA/SARIMA/SARIMAX model attempts with **focus on performance, complexity, and improvement** — not repeating what the raw ARIMA logs already tell.

---

### 🔍 **Overall Goal:**

You're trying to forecast the `Close` price, and optimizing for **low RMSE / MAPE** with a **good model fit**.

---

### ✅ **Summary of Model Attempts**

| Attempt | Model Type     | Data Size | Model Order                      | RMSE       | MAPE      | AIC    | Comments                                                         |
| ------- | -------------- | --------- | -------------------------------- | ---------- | --------- | ------ | ---------------------------------------------------------------- |
| 1       | ARIMA          | 2086      | (1,0,0)                          | —          | —         | 23998  | Weak. No differencing, seasonal pattern ignored.                 |
| 2       | ARIMA          | 1500      | (0,1,1)                          | —          | —         | 17668  | Improved over (1,0,0); captures trend better.                    |
| 3       | SARIMA         | 1500      | (1,1,0)(0,0,0)[7]                | 1162.53    | 31.81%    | ~17668 | Similar to ARIMA (0,1,1), but with very minor structural change. |
| 4       | SARIMAX        | 1500      | (0,1,1)(0,1,1)[7]                | 829.92     | 21.71%    | 17629  | Great improvement—seasonality captured.                          |
| 5       | SARIMAX + exog | 1500      | (1,0,1)(1,0,1)[7] + exog vars    | **295.77** | **7.36%** | 18230  | Best performance, despite higher AIC; exogenous vars help a lot. |
| 6       | SARIMAX + exog | 1200      | (2,0,[1,2])(2,0,[1,2])[7] + exog | **259.86** | **6.51%** | 16337  | Even better; reduced data = less noise + better generalization.  |

---

### 🎯 **Observations**

- **ARIMA alone is weak**, especially without differencing (`d=0`), and no seasonal handling.
- **Basic SARIMA helps**, but only modestly.
- **SARIMAX models (with exogenous features like Volume, MA30, MA90)** drastically improve performance.
- Going from 1500 to 1200 obs actually **reduced error** — possible overfitting avoided.
- Last model (Attempt 6) is the **sweet spot**: lowest RMSE & MAPE, logical structure, seasonal dynamics and external regressors fully used.

---

### ⚙️ **Next Best Actions?**

1. Try using **cross-validation** (like time series split) to validate your best model stability.
2. Plot **residuals + predictions** side-by-side with actuals to visually confirm reliability.
3. If you want to further refine:
   - Test for **stationarity & differencing need** again.
   - Tune SARIMAX with auto_arima or GridSearchCV over p,d,q,P,D,Q,S.
   - Consider **XGBoost or LSTM** as nonlinear alternatives for comparison.

---

Let me know if you want visualizations next or a final code export of the best model (6th one).

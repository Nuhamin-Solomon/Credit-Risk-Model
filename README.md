# Credit-Risk-Model
Credit Scoring Business Understanding
How Basel II’s emphasis on risk measurement influences our need for an interpretable and well-documented model
Basel II requires banks to measure risk precisely and justify capital allocation decisions. This means models must be explainable, auditable, and reproducible. Documentation of data sources, features, transformations, and validation results is crucial.

Why we need a proxy target (and associated business risks)
Since the dataset lacks a direct "default" label, we create a proxy (e.g., disengaged customers via RFM clustering) as a high-risk indicator. Risks include label validity, selection bias, regulatory challenges, and operational mistakes. Mitigation includes validating the proxy, documenting assumptions, and implementing guardrails.

Key trade-offs: simple interpretable model vs complex high-performance model
Logistic Regression + WoE: Highly interpretable, easier regulatory acceptance, robust with small datasets.
Gradient Boosting: Higher predictive power, captures non-linearities, requires extra explainability and monitoring.
Recommendation: Start with Logistic+WoE for regulatory compliance and transparency, while optionally testing Gradient Boosting experimentally.
# AI Usage Log

## Tools Utilized
* **Claude (Anthropic)**: Primary tool used for architectural design, code generation, and conceptual troubleshooting throughout the project lifecycle.

---

## Technical Implementation & Learning
The following table outlines the specific tasks delegated to AI and the resulting technical insights gained.

| Task | AI Tool | Implementation Details | Key Takeaways & Logic |
| :--- | :--- | :--- | :--- |
| **ETL Pipeline Design** | Claude | Architected modular Extract, Transform, and Load functions. | Learned the necessity of **rolling windows** for technical indicators to prevent data leakage and ensure feature consistency. |
| **Feature Engineering** | Claude | Generated scripts for SMA, RSI, EMA, and Volatility indicators. | Understood the mathematical intuition of each indicator and the impact of hyperparameter (window size) selection on model sensitivity. |
| **Model Training** | Claude | Implemented Logistic Regression with standardized features and proper splitting. | Learned why `shuffle=False` is mandatory for **time-series validation** to avoid training on future data (look-ahead bias). |
| **API Integration** | Claude | Developed the `PySimFin` class wrapper for data acquisition. | Mastered **Object-Oriented Programming (OOP)** principles, specifically encapsulation, error handling, and managing API rate limits. |
| **Streamlit Interface** | Claude | Scaffolded a multi-page dashboard for data visualization. | Learned state management in Streamlit and the implementation of `st.cache` to optimize performance for data-heavy operations. |
| **System Debugging** | Claude | Resolved path-related errors and data schema mismatches. | Improved proficiency in interpreting Python tracebacks and managing relative file paths in modular project structures. |

---

## Reflection & Accountability

### Understanding the Logic
In accordance with the project's AI Usage Policy, every line of generated code was audited and verified. For instance, the transition from standard cross-validation to a temporal split was a critical architectural decision; I have ensured I can explain the underlying logic of time-dependent probability in the context of stock prediction for the in-class Q&A.

### Strategic Use of AI
The use of Claude served as a force multiplier for boilerplate tasks (such as Streamlit scaffolding and basic HTTP request structures). This efficiency allowed more bandwidth to be dedicated to **feature selection**, **hyperparameter tuning**, and **result evaluation**.

### Risk Mitigation
To avoid "black box" dependency, I manually added comprehensive documentation to all AI-assisted functions. This process confirmed that I do not merely possess the code, but I fully understand the logic governing the ETL and Machine Learning workflows. 

---
*This log serves as a record of reflective practice and technical development as required by the course guidelines.*

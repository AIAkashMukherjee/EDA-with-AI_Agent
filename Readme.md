# 🧠 AI Agent for EDA (Exploratory Data Analysis)

An interactive, AI-powered application that helps users upload a CSV dataset and automatically generates:

- Data preview & schema
- Missing value summary
- Visualizations (histograms, boxplots, violin plots, heatmaps, etc.)
- AI-generated insights (Gemini )

> **Note**: Gemini API key required for AI insights.

---

## 🚀 Features

- 📊 Auto-generates EDA visuals
- 🔍 Detects missing values
- 🤖 Optional AI-driven analysis using Gemini or ChatGPT
- 🖼️ Violin plots, boxplots, histograms, correlation heatmap
- 📁 Upload and analyze any CSV file

---

## 📂 Project Structure

```bash
.
├── app.py                # Main application file
├── templates/
│   └── index.html        # Frontend upload & results UI
├── static/
│   └── images/           # Stores generated plots
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

# Example Output

* Schema & Preview
* Missing Values Summary
* Histograms
* Boxplots
* Violin Plot
* Correlation Heatmap
* AI-generated Insight Summary (if enabled)

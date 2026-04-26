# 🌍 Neo-Romance Evolution Simulator

This project uses Genetic Algorithms (GA) and NLP to simulate the linguistic evolution from Latin to Romance languages, creating a synthesized **"Neo-Romance"** language.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit Interface:**
   ```bash
   streamlit run src/app.py
   ```

## 📂 Project Structure

- `src/app.py`: Streamlit web interface with model controls.
- `src/translator.py`: Core translation logic combining GA rules and grammar.
- `src/precompute_neo.py`: Logic for running the Genetic Algorithm to evolve the dictionary.
- `src/genetics.py`: Genetic Algorithm engine for phonetic drift.
- `src/config.py`: Centralized path and constant management.
- `data/`: 
    - `neo_romance_dict.json`: (Included) Pre-computed evolutionary results.
    - `lexicon.csv`: (Local only) Source etymological data for training.

## 🛠️ New Workflow: On-Demand Evolution

The app now supports dynamic dictionary regeneration:
1. **Translation:** Use the main text area to translate Latin sentences into Neo-Romance instantly using the pre-computed dictionary.
2. **Model Regeneration:** If you have `lexicon.csv` locally, use the **"Re-processar Dicionário (AG)"** button in the sidebar to re-run the Genetic Algorithm.
3. **Dynamic Updates:** The system will evolve 500+ Latin roots based on current phonetic rules and refresh the app with the new synthesized vocabulary.

## ☁️ Deployment

This app is optimized for **Streamlit Cloud**:
- The pre-computed dictionary is included for immediate functionality.
- Regeneration features are automatically disabled in cloud environments where source data is restricted, ensuring stability.

---
*Created with Gemini CLI*

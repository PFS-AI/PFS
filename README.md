<div align="center">
  <img src="./static/logo.png" alt="Logo" width="150" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px;"/>
  <h1>Precision File Search (PFS)</h1>
  <a href="https://pfs-ai.github.io/PFS/" target="_blank">Project Official Website</a>
<p><strong>An AI-powered, locally-run platform that transforms digital chaos into organized clarity.</strong></p>

 <p>
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/Framework-FastAPI-green.svg" alt="Framework FastAPI">
    <img src="https://img.shields.io/badge/LangChain-Orchestration-blueviolet" alt="LangChain">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow" alt="Hugging Face">
    <img src="https://img.shields.io/badge/Scikit--learn-Classifier-orange" alt="Scikit-learn">
    <img src="https://img.shields.io/badge/Qdrant-Vector_DB-953da1" alt="Qdrant">
    <br>
	<img src="https://img.shields.io/badge/License-MPL_2.0-blue.svg" alt="License: MPL 2.0">
	<img src="https://img.shields.io/github/issues/PFS-AI/PFS" alt="GitHub issues">
	<img src="https://img.shields.io/github/discussions/PFS-AI/PFS" alt="GitHub discussions">
	 <br>
    <a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=alert_status" alt="Quality Gate Status"></a>
    <a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=security_rating" alt="Security Rating"></a>
    <a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=sqale_rating" alt="Maintainability Rating"></a>
    <a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=vulnerabilities" alt="Vulnerabilities"></a>
</p>

</div>

<div align="center">
  <img src="./static/demo/demo.gif" alt="Precision File Search Demo" width="850" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
</div>

---

### **Table of Contents**

1.  [**🌟 About Precision File Search (PFS)**](#-about-precision-file-search-pfs)
    -   [🎯 The Problem it Solves](#-the-problem-it-solves)
    -   [🎯 Who is This For?](#-who-is-this-for)
    -   [Core Features](#core-features)
2.  [**🚀 Getting Started**](#-getting-started)
    -   [Prerequisites](#prerequisites)
    -   [🔧 Installation](#-installation)
    -   [⚙️ Configuration (First Run)](#️-configuration-first-run)
    -   [✨ Key Features](#-key-features)
2.  [**🚀 Getting Started**](#-getting-started)
    -   [Prerequisites](#prerequisites)
    -   [🔧 Installation](#-installation)
    -   [⚙️ Configuration (First Run)](#️-configuration-first-run)
3.  [**📖 How to Use**](#-how-to-use)
4.  [**📚 Full Documentation**](#-full-documentation)
5.  [**🏗️ Architecture & Technology Stack**](#️-architecture--technology-stack)
6.  [**🧪 Request for Testing & Feedback**](#-request-for-testing--feedback)
7.  [**🤝 How to Contribute**](#-how-to-contribute)
8.  [**🛡️ Security First**](#️-security-first)
9.  [**🙏 Acknowledgments**](#-acknowledgments)
10. [**📝 License**](#-license)
11. [**📜 Code of Conduct**](#-code-of-conduct)
12. [**🏢 Commercial Development & Enterprise Support**](#-commercial-development--enterprise-support)

---

## 🌟 About Precision File Search (PFS)

**Precision File Search (PFS)** is a powerful, desktop file search and management application that goes far beyond simple keyword matching. It integrates a classic high-speed search engine, a trainable machine learning classifier, and a state-of-the-art AI-powered RAG (Retrieval-Augmented Generation) pipeline into a single, intuitive user interface.

This project is designed for everyone, from casual users needing to find a lost document to developers searching through codebases and researchers extracting insights from vast archives. It runs entirely on your local machine, ensuring your data remains private and secure.

### 🎯 The Problem it Solves

In a world of digital clutter, finding the right information quickly is a constant challenge. Standard operating system searches are often slow or limited to filenames. PFS addresses these pain points by providing a multi-layered solution:

-   **Find anything, instantly:** Locate files and folders by name, content, or even abstract concepts.
-   **Bring order to chaos:** Automatically sort thousands of scattered documents into logical categories with a single click.
-   **Get answers, not just links:** Ask complex questions in natural language and receive synthesized, summarized answers directly from the content of your files.

---

### 🎯 Who is This For?

PFS is designed for anyone who needs to quickly find and make sense of information stored on their local machine. If you work with a large volume of digital documents, this tool is for you.

*   **Researchers, Academics, and Students:** Instantly search through thousands of research papers, articles, and notes. Go beyond keyword search to find conceptual links between documents and get AI-powered summaries of complex topics.

*   **Developers and Engineers:** Quickly search entire codebases with high-speed regex and content matching. Find function definitions, configuration files, or error logs in seconds without leaving your workflow.

*   **Legal and Business Professionals:** Sift through case files, contracts, and financial reports with ease. Ask natural language questions like, *"Find all contracts with a termination clause"* and get direct answers from your documents.

*   **Writers and Content Creators:** Rediscover lost ideas and sources from your personal archive. Effortlessly locate notes, drafts, and reference materials based on the concepts within them.

*   **Anyone Drowning in Digital Clutter:** If your "Documents" folder is a chaotic mix of files, PFS is your solution. Use the classifier to automatically sort files into logical folders and finally find what you need, when you need it.

---

## Core Features

-   🚀 **High-Performance File Engine:** A foundational layer for **high-speed**, precise queries using **regex** and advanced filters. The engine is built for maximum efficiency, using a **concurrent architecture** to parallelize file system operations and keep memory usage low, even when searching massive files. It supports **fast searching** in over **60** file types and can perform **~3,000** file content searches per second.

-   🧠 **Intelligent Semantic Search (RAG):** Transforms your local files into a searchable knowledge base, allowing you to search by **meaning** and **context**, not just keywords. It features an **intelligent re-indexing system** that tracks file changes, preventing redundant processing and ensuring that updates to the knowledge base are both fast and efficient.

-   ✨ **AI-Powered Orchestrator:** The top-level intelligence that allows you to converse with your data. It understands **natural language**, routes queries to the best search tool (classic or semantic), and synthesizes results into direct, actionable answers and summaries.

-   🤖 **Trainable ML Document Classifier:** A built-in **Machine Learning** model that brings order to chaos by automatically sorting documents into logical categories. You can re-train the model with your own data for custom classification.

-   🔒 **Privacy-First Architecture:** The entire application, including the AI models and vector database, runs **100% locally** on your machine. Your files are never uploaded, and your data never leaves your computer.

---

<div align="center">
  <img src="./static/demo/1.png" alt="Screenshot of the AI Search interface showing a user interface" width="800" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
</div>
<div align="center">
  <img src="./static/demo/2.png" alt="Screenshot of the AI Search interface showing a user interface" width="800" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
</div>
<div align="center">
  <img src="./static/demo/3.png" alt="Screenshot of the AI Search interface showing a user interface" width="800" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
</div>
<div align="center">
  <img src="./static/demo/4.png" alt="Screenshot of the AI Search interface showing a user interface" width="800" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
</div>

### 🚀 A Note on This Version

While this initial version provides a powerful and complete user experience, the roadmap for evolving it into a fully production-ready application includes several key enhancements. The current system is highly functional, but the following planned improvements will focus on scalability, ease of deployment, and long-term robustness.

**Planned Enhancements:**

*   **Deployment and Packaging:** Creating standalone, easy-to-use installers for major operating systems (e.g., `.exe` for Windows, `.dmg` for macOS) to eliminate the need for manual setup for non-technical users.

*   **Robustness and Testing:** Building a comprehensive suite of **unit and integration tests** to ensure long-term stability, verify that each component functions correctly, and prevent regressions during future development.

This is the initial version of the project, and there is a huge potential for improvement. Future development could include adding new functionalities, refining existing features, and extending the platform's capabilities for production environments.

---

## 🚀 Getting Started

This guide provides instructions for both regular users (via a simple installer) and developers (running from source code).

### Prerequisites

*   **Python 3.11 or newer** (only required for developers running from source).
*   An internet connection is required during the initial setup to download dependencies and models.

*   **Hardware Considerations:**
    *   **Base Application:** PFS is designed to be efficient. The core platform (Classic Search, Classifier, Semantic Indexing) runs well on most modern consumer hardware, including systems with CPUs up to 5 years old and a **minimum of 8GB of RAM**.
    *   **For Local AI Models:** To use the **AI Search** feature with a locally-run Large Language Model (LLM) (e.g., via Ollama, LM Studio), a dedicated GPU is strongly recommended for a smooth experience.
        *   A **minimum of 4GB of VRAM** (GPU memory) is needed for smaller models.
        *   Larger, more capable models will require significantly more VRAM (e.g., 8GB, 16GB, or more).
    *   *Note: These GPU requirements **do not apply** if you are using a cloud-based LLM provider (like OpenAI, Groq, etc.) via an API key, as the computation happens on their servers.*

---

### For Users (Recommended Easy Installation)

The easiest way to install Precision File Search is to download the latest official installer for Windows.

[![Latest Release](https://img.shields.io/badge/Download-V1.1.1-blueviolet?style=for-the-badge)](https://github.com/PFS-AI/PFS/releases/latest)

1.  Click the button above to go to the latest release page.
2.  Under the **Assets** section, download the `PFS-SETUP_vX.X.X.exe` file.
3.  Run the installer and follow the on-screen instructions.
4.  Once installed, launch the application and navigate to the **Settings** tab to configure your LLM and Semantic models as needed.

## 🧠 Precision File Search: Offline AI Inference Setup with Ollama
**Powered by IBM Granite 4.0 Hybrid Intelligence**

![IBM Co-Branding](https://www.ibm.com/brand/experience-guides/developer/static/b2467f8258b1b1b99aecfa46fac8976b/3cbba/10_co-branding-events.png)

Precision File Search (PFS) supports **OpenAI-compatible URLs** for AI inference. These endpoints can be hosted by:

- 🌐 Online providers (e.g., OpenAI, Together.ai)
- 🖥️ Offline inference engines like **Ollama**, **LM Studio**, or **Docker-based LLMs**

For offline use, we recommend **Ollama** for its simplicity, speed, and compatibility.

### ⚙️ How to Use Ollama for Offline AI Inference

1. **Download Ollama**
   👉 [https://ollama.com/download](https://ollama.com/download)

2. **Run the IBM Granite Model**
   Open PowerShell or Command Prompt and enter:
   ```powershell
   ollama run granite4:micro-h
   ```
   This will download and launch **IBM Granite 4.0 Micro-Hybrid**, a state-of-the-art hybrid LLM optimized for PFS.

3. **Configure PFS to Use Ollama**
   - Open the **PFS Settings** page
   - In the **API Key** field, enter:
     ```
     ollama
     ```
   - In the **Model Name** field, enter:
     ```
     granite4:micro-h
     ```

4. **Apply Settings**
   Save your changes, close PFS, and relaunch the application.

5. **Enjoy Offline Precision**
   You now have a fully offline Precision File Search experience powered by IBM-grade intelligence.

---

### For Developers (Running from Source)

<details>
<summary>Click here for instructions on how to run the project from the source code.</summary>

#### 🔧 Installation

1.  **Clone the Repository:**
    Open your terminal and clone the PFS repository:
    ```bash
    git clone https://github.com/PFS-AI/PFS.git
    cd PFS
    ```

2.  **Create a Virtual Environment:**
    Using a virtual environment is crucial for isolating the project's dependencies and ensuring a clean, reproducible setup.
    ```bash
    # Create the virtual environment
    python -m venv .venv

    # Activate it
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    This project uses the modern `pyproject.toml` file to manage all dependencies. Install them using `pip`, which is included with Python.
    ```bash
    pip install -e .
    ```
    *   **What this command does:** `pip` reads `pyproject.toml` to find and install all required libraries. The `-e` flag ("editable" mode) installs the project so that changes you make to the code are immediately effective, which is ideal for development.
    *   **Be patient:** The initial installation may take several minutes as it downloads large AI and machine learning libraries.

    <details>
    <summary><i>Our Approach to Dependency Management (pip vs. uv)</i></summary>

    For a project of this complexity, choosing the right dependency management strategy is key. We use **`pyproject.toml`** as our standard definition file, which can be installed with the official **`pip`** installer or the high-speed alternative, **`uv`**.

    This separates the **"what"** (the list of dependencies in `pyproject.toml`) from the **"how"** (the tool used to install them).

    While the command above uses `pip` for universal compatibility, advanced users can get a significant speed boost by using `uv`. Since `uv` also understands the `pyproject.toml` standard, the command is nearly identical:

    ```bash
    # Optional: Using uv for a much faster installation
    pip install uv
    uv pip install -e .
    ```
    </details>

4.  **Set Up LangSmith Tracing (Optional but Recommended):**
    This project is integrated with [LangSmith](https://smith.langchain.com/) for tracing and debugging the AI-powered search pipelines. To enable it:

    *   In the project's root directory, find the file named `.env.example`.
    *   **Create a copy** of this file and **rename the copy to `.env`**.
    *   Go to the [LangSmith website](https://smith.langchain.com/), sign up, and create an API key.
    *   Open your new `.env` file and paste your API key.

    Your `.env` file should look like this:
    ```env
    # This file enables LangSmith tracing and is NOT committed to Git.

    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_PROJECT="Precision-File-Search"
    LANGCHAIN_API_KEY="lsv2_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```
    *If you skip this step, the application will still work perfectly, but you won't get detailed tracing in your LangSmith dashboard.*

5.  **Run the Application:**
    Start the backend server with the following command:
    ```bash
    python main.py
    ```
    To start the backend server with more detailed logging:
    ```bash
    python main.py --debug
    ```
    *   The server will start, and on the **first run**, it will begin downloading the default embedding and reranker models. You will see progress bars in your terminal. This can take several minutes.
    *   Once the server and model checks are complete, your default web browser will automatically open to `http://127.0.0.1:9090`.

</details>


### ⚙️ Configuration (First Run)

The application is usable out-of-the-box for Classic Search and the Document Classifier. **To enable the Semantic Search and AI Search features, you must configure the necessary models.**

All models are downloaded automatically from Hugging Face and cached locally the first time they are needed.

#### 1. LLM Settings (for AI Search Summarization)

This model is responsible for understanding natural language, routing queries, and summarizing search results.

-   **Navigate to the Settings Tab:** In the web UI, click on the "Settings" tab.
-   **Enter LLM Credentials:** Scroll to the "LLM Settings" section. You need to provide:
    -   **API Key:** Your secret API key.
    -   **Model Name:** The identifier for the model you want to use (e.g., `meta-llama/llama-4-maverick-17b-128e-instruct`).
    -   **API Base URL:** The endpoint for the API. This is crucial for connecting to services like Groq, or local LLMs like Ollama.
-   **Save Settings:** Click the "SAVE ALL SETTINGS" button.

#### 2. Semantic Model Settings (for Semantic Search)

These models power the ability to search by meaning and context.

-   **Embedding Model (Required for Semantic Search)**
    -   **Purpose:** This model converts your documents and search queries into numerical vectors, which allows the system to find matches based on semantic similarity.
    -   **How to Choose:** For most users, a light, fast, and effective model is recommended. When selecting a model from the [Hugging Face MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), look for models that balance performance and size.
    -   **Recommendation:** A great starting point is `all-MiniLM-L6-v2`. It's relatively small, fast, but not supports multiple languages well.
    -   **Action:** Enter the full Hugging Face model identifier (e.g., `sentence-transformers/all-MiniLM-L6-v2`) in the `Embedding Model` field in Settings. The format must be `publisher/model-name`.

-   **Reranker Model (Optional, for Advanced Users)**
    -   **Purpose:** The Reranker (or Cross-Encoder) dramatically improves the relevance of search results by performing a second, more computationally intensive analysis on the top results from the embedding search.
    -   **Trade-offs:** This provides higher accuracy at the cost of slower search speeds and requires more system resources (especially VRAM). It is recommended for

> **Important Note on Models & Connectivity:**
> Hugging Face models can be large, with sizes ranging from **~100 MB to several gigabytes**. An active internet connection is required the first time you run the application with a new model, as it will be downloaded and cached locally. This process is for downloading model files only; **your personal data and file content are never sent to the internet.**

---

### 📖 How to Use

Precision File Search offers four powerful, interconnected tools. Start with AI Search for general queries, or use a specific tool for a targeted task.

#### 🤖 **AI Search: Your Conversational Starting Point**
This is your primary interface. Type a question or command in plain English, and let the AI orchestrator choose the best search strategy and deliver a summarized answer.

*   ***Example Query:*** `"Find my report on Q3 earnings and summarize the key findings."`
*   ***Example Meta-Query:*** `"PFS: how does the reranker work?"` (Use the `PFS:` prefix to ask about the application itself).

#### 🧠 **Semantic Search: Find by Meaning, Not Keywords**
Use this for deep, conceptual searches when you know *what* you're looking for but not the exact words. This requires a one-time setup for each folder.

1.  Go to the **"Semantic"** tab.
2.  Provide a path to your documents and click **`BUILD / UPDATE INDEX`**.
3.  Once complete, ask questions like: `"what were the main conclusions of the market analysis?"` to find the most relevant document chunks.

#### ⚡ **Classic Search: High-Speed, Precise Queries**
Perfect for instant searches that **do not require any indexing**. Use it for folders that change frequently or for finding specific file names and content on-the-fly.

*   **Find keywords inside files:** Scans file content directly.
*   **Locate specific file or folder names:** Supports wildcards (e.g., `invoice_*.pdf`).
*   **Filter by category, size, or date:** Narrow down results with powerful filters.

#### 🗂️ **Classifier: Bring Order to Chaos**
Automatically sort a messy folder full of mixed documents into organized subdirectories.

1.  Go to the **"Classifier"** tab.
2.  Enter the path to the cluttered folder and click **`START CLASSIFICATION`**.
3.  Once the analysis is finished, use the **`Auto Organize`** button for one-click sorting into neatly categorized subfolders.

---

## 📚 Full Documentation

This project has a full documentation that covers every topic to start from scratch to advanced. The complete guide is built directly into the platform.

Once the application is running, the documentation is available using the platform in the About tab and at the following address:

**[http://127.0.0.1:9090/documentation](http://127.0.0.1:9090/documentation)**

<div align="center">
  <img src="./static/demo/5.png" alt="Precision File Search Documentation" width="800" style="border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
</div>

---

## 🏗️ Architecture & Technology Stack

PFS is built with a modern, modular architecture to ensure performance, maintainability, and security.

-   **Backend:**
    -   **Framework:** FastAPI
    -   **AI/ML Orchestration:** LangChain
    -   **NLP Models:** Hugging Face Transformers, Sentence-Transformers
    -   **Machine Learning:** Scikit-learn
    -   **Vector Database:** Qdrant (local on-disk)
    -   **Data Storage:** SQLite (for document store & configuration)
    -   **File Processing:** Unstructured.io

-   **Frontend:**
    -   Vanilla HTML5, CSS3, and JavaScript (ES Modules)
    -   **UI Libraries:** Font Awesome (icons), Marked.js (Markdown rendering), DOMPurify (security)

The backend exposes a RESTful API and a WebSocket for real-time communication. The AI orchestrator (`ai_search.py`) acts as the central brain, delegating tasks to specialized modules for semantic retrieval, classic search, and LLM interaction.

---

## 🧪 Request for Testing & Feedback

 We kindly request that you test the application and report any issues or suggestions.

-   **Report Bugs:** If you encounter a bug, an error, or unexpected behavior, please [**open an issue**](https://github.com/PFS-AI/PFS/issues) on GitHub. Include steps to reproduce the problem and any relevant logs from the terminal.
-   **Suggest Features:** Have an idea for a new feature or an improvement to an existing one? We'd love to hear it! Please [**start a discussion**](https://github.com/PFS-AI/PFS/discussions) to share your thoughts.

---

## 🤝 How to Contribute

We are excited to welcome contributions from the community! Whether it's reporting a bug, improving translations, suggesting a feature, or writing code, your help is greatly appreciated.

### Types of Contributions We're Looking For

*   **Code Contributions:** Fixing bugs or implementing new features.
*   **Documentation:** Improving the README, documentation pages, or inline code comments.
*   **Bug Reports & Feature Requests:** Submitting detailed issues and well-thought-out ideas in the [Issues tab](https://github.com/PFS-AI/PFS/issues).


### 🌍 A Special Call for Translators

The internationalization of this project is a key priority. To make the application accessible to a global audience, initial translations for **18 languages** have been provided:

-   Arabic (`ar.json`)
-   Armenian (`hy.json`)
-   Chinese (`ch.json`)
-   English (`en.json`)
-   French (`fr.json`)
-   Georgian (`ka.json`)
-   German (`de.json`)
-   Hindi (`hi.json`)
-   Italian (`it.json`)
-   Japanese (`jp.json`)
-   Korean (`kr.json`)
-   Persian (`fa.json`)
-   Romanian (`ro.json`)
-   Russian (`ru.json`)
-   Spanish (`es.json`)
-   Turkish (`tr.json`)
-   Ukrainian (`uk.json`)
-   Urdu (`ur.json`)

While these translations are a great start, they may not be perfect. We rely on native speakers to help us ensure the UI is clear, natural, and accurate in every language.

**We would be incredibly grateful if you could check the translation for your native language.**

**How to Help with Translations:**
1.  Navigate to the `static/lang/` directory in this repository.
2.  Find the JSON file corresponding to your language from the list above (e.g., `es.json` for Spanish, `de.json` for German).
3.  Review the text values for any grammatical errors, awkward phrasing, or incorrect terminology.
4.  If you find something to improve, please [**open a new GitHub Issue**](https://github.com/PFS-AI/PFS/issues/new). In the issue, please include:
    *   The language file name (e.g., `de.json`).
    *   The specific key (e.g., `"appTitleAdvanced"`).
    *   The incorrect text and your suggested improvement.

Your contribution here is invaluable for making the application accessible to a global audience!

---

### General Contribution Workflow

To ensure a smooth and collaborative process for code changes, we have a simple guideline:

**➡️ Please discuss your ideas in a GitHub Discussion *before* starting to write code.**

This approach helps us:
-   **Align on Goals:** Ensure your proposed change fits with the project's vision and roadmap.
-   **Avoid Duplicate Work:** Check if someone else is already working on a similar feature.
-   **Refine the Technical Approach:** Discuss the best way to implement your idea and get early feedback.
-   **Streamline the Review Process:** Make the pull request review much faster and more straightforward for everyone.

**Workflow Steps:**

1.  **Start a Discussion:** Go to the [**Discussions tab**](https://github.com/PFS-AI/PFS/discussions) and open a new topic. Clearly describe the bug you want to fix or the feature you want to add. We'll work with you to define the scope and plan.

2.  **Fork & Branch:** Once the idea is discussed and agreed upon, fork the repository and create a new branch for your work.
    ```bash
    git checkout -b feature/your-amazing-feature
    ```

3.  **Develop & Test:** Make your changes, adhering to the project's coding style. Make sure to test your changes thoroughly.

4.  **Submit a Pull Request:** Push your branch to your fork and open a pull request against the `development` branch of the PFS repository. Please provide a clear description of your changes and link to the original discussion topic.

We look forward to collaborating with you!

---


## 🛡️ Security First

Your data privacy and system integrity are our top priorities. PFS is engineered with a security-first mindset. Throughout the development process, best practices in AI security were implemented to safeguard against emerging threats.

To validate our commitment to high-quality, secure code, the project is continuously analyzed by **SonarCloud**, a leading static code analysis tool. We are proud to have achieved a **passed Quality Gate** and maintain an **'A' rating for both Security and Maintainability**, ensuring the code is not only functional but also robust and secure from the ground up.

<a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=alert_status" alt="Quality Gate Status"></a> <a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=security_rating" alt="Security Rating"></a> <a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=sqale_rating" alt="Maintainability Rating"></a> <a href="https://sonarcloud.io/summary/new_code?id=Eng-AliKazemi_PFS"><img src="https://sonarcloud.io/api/project_badges/measure?project=Eng-AliKazemi_PFS&metric=vulnerabilities" alt="Vulnerabilities"></a>

**Security Measures:**
-   **Local Execution:** The server is bound to `127.0.0.1`, making it inaccessible from the network.
-   **No Data Exfiltration:** Your files are never uploaded or exposed to the internet.
-   **Path Traversal Prevention**: Before any file system operation, the application validates and canonicalizes the target path. This process safely resolves any relative path components (like . or ..), effectively blocking any attempt to access unintended directories. This ensures that only the explicitly specified path is ever accessed, providing robust security against path traversal attacks while still allowing you the flexibility to search any folder on your system.
-   **SQL Injection Prevention:** All database queries use parameterized statements, an industry-standard technique that separates commands from data. This makes it impossible for malicious user input to be executed as a database command.
-   **Prompt Injection Hardening:** Prompts are engineered with defensive instructions that reinforce the AI's core mission and warn it about untrusted content. This helps the AI resist malicious instructions hidden in files or user queries.
-   **ReDoS Mitigation:** Regular expression matching is run in a separate, non-blocking process. This isolates the operation, ensuring a malicious pattern cannot cause a denial-of-service attack that would freeze the application.
-   **XSS Prevention:** All user-facing output, especially from the AI, is sanitized to prevent cross-site scripting attacks.

**Note on Enterprise Security:** The default configuration is optimized for individual use and ease of installation, using libraries like SQLite and local on-disk Qdrant. For enterprise-level security, the platform has the flexibility to be extended; for example, by migrating from SQLite to a server-based database and from local Qdrant to its server version. Advanced encryption features are planned for a future release, but the current architecture will be preserved as the default to ensure the public project remains easy to install and use.

For more details, please see the full security documentation.

---

## 🙏 Acknowledgments

The development of Precision File Search relies on the exceptional work of the open-source community. We extend our deepest gratitude to the developers behind the incredible tools that made PFS possible, including **IBM Granite Team**, FastAPI, LangChain, Hugging Face, Scikit-learn, Qdrant, and many more.

For a complete list of third-party dependencies and their licenses, please see the [DEPENDENCIES.md](DEPENDENCIES.md) file.

---

### 📄 License

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.

You are free to use, study, share, and modify the software. If you modify any MPL-licensed files, you must make the source code of those specific files available. You may combine this software with proprietary code in a larger project without needing to release the source code of your other components.

**🔔 Additional Attribution Requirement**
In accordance with MPL 2.0, all distributions of this software in **binary and source form** must include the following user-visible attribution in at least one of the following locations:
- In the README
- On a startup/splash screen
- In an "About" or "Acknowledgements" dialog box
- In the primary documentation or "Help" menu
- As a footer on a command-line tool's initial output

**Required attribution text:**
> *Powered by Precision File Search (PFS) from https://github.com/PFS-AI.*

For the full license text, please see the [LICENSE](LICENSE) file.

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by the [PFS Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## 🏢 Commercial Development & Enterprise Support

While PFS is a powerful tool for private use, we understand that businesses and professional organizations may require features, integrations, or support beyond the scope of the open-source project.

Professional services are available for organizations that need to take PFS to the next level. Please reach out to discuss your requirements for:

-   **Custom Feature Development:** Tailoring PFS to meet your specific business logic and workflow needs.
-   **Enterprise Integrations:** Integrating PFS with your existing systems, such as SSO (Single Sign-On), cloud storage, or internal databases.
-   **Priority Support & Maintenance:** Securing dedicated support contracts for bug fixes, assistance, and maintenance.
-   **On-Premise Deployment & Consulting:** Professional assistance with deploying and scaling PFS in a secure, private corporate environment.
-   **AI/RAG Pipeline Consulting:** Leveraging the expertise behind PFS to help you build similar intelligent search and data processing solutions for your own applications.

For all commercial inquiries, please contact the project lead, **Ali Kazemi**. He is a certified **IBM AI Engineer** and **AI Solution Architect** who specializes in developing and consulting on intelligent systems for enterprise clients.

He holds more than 20 specialization and professional certifications from **IBM**, underscoring his deep expertise in AI and related technologies, and is a member of the **European AI Alliance**.

![European AI Alliance](https://futurium.ec.europa.eu/system/files/styles/fut_group_logo/private/2025-09/AI%20Alliance%20logo%201.png?h=d0ac640d&itok=C2EbFBbk)

🔗 [Read the official project article on the European AI Alliance portal](https://futurium.ec.europa.eu/en/apply-ai-alliance/community-content/precision-file-search-precise-combination-search-engine-and-ai)

#### Contact & Inquiries

<a href="https://linkedin.com/in/e-a-k" target="_blank"><img src="https://img.shields.io/badge/Connect-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=555" alt="Connect on LinkedIn"/></a>

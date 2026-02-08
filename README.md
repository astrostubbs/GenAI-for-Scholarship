# Generative AI for Scholarship

**Harvard Data Science Initiative (HDSI) & Faculty of Arts and Sciences (FAS)**

🌐 **Course Website:** [https://astrostubbs.github.io/GenAI-for-Scholarship/](https://astrostubbs.github.io/GenAI-for-Scholarship/)

---

## About This Series

This three-part introductory workshop series provides a hands-on introduction to generative AI tools for STEM researchers.

**Note:** These three introductory sessions will be followed by five additional sessions after Spring break, covering advanced topics and specialized applications.

All materials — notebooks, data files, and resource pages — are available in this repository for use during and after the sessions.

**Please bring your laptop to all sessions.** These are hands-on workshops where you will be working directly with AI tools and code.

---

## Sessions

### Week 1 — The Basics
**Friday, February 20, 2026 · 4:00–5:30 pm · Northwest Building, Room B103**

Introduction to Google's Gemini AI toolkit: Gemini, NotebookLM, and Gems. Effective prompting and responsible AI use in research.

**Prerequisites:** No prior AI experience required. This session provides a basic introduction to using the Google AI toolkit in conjunction with uploaded files and custom prompts. Participants need a Harvard-affiliated Google account, such as a g.harvard.edu email address.

📄 **Materials:**
- [Session 1 Page](https://astrostubbs.github.io/GenAI-for-Scholarship/session1-foundation.html)
- [Session 1 Exercises](https://astrostubbs.github.io/GenAI-for-Scholarship/exercise-session1.html)

---

### Week 2 — The AI-Empowered Coder
**Friday, February 27, 2026 · 4:00–5:30 pm · Northwest Building, Room B103**

Incorporating AI into Python workflows: code generation, revision, and debugging.

**Prerequisites:** Prior experience with Python notebooks required. Students should have a Colab folder in their Google Drive.

📄 **Materials:**
- [Session 2 Page](https://astrostubbs.github.io/GenAI-for-Scholarship/session2-coder.html)
- [Demonstration Notebook](notebooks/session2/ai_python_demo.ipynb) - AI integration with Colab, local Jupyter, and Harvard RC cluster

---

### Week 3 — Unleashing Claude Code Command Line Interface as a Problem Solver
**Friday, March 6, 2026 · 4:00–5:30 pm · Northwest Building, Room B103**

Hands-on with Claude Code: agentic AI programming and data analysis from the command line.

**Prerequisites:** Students should be comfortable with command line (terminal) interactions with the Mac operating system, with Python, and with quantitative data analysis.

📄 **Materials:**
- [Session 3 Page](https://astrostubbs.github.io/GenAI-for-Scholarship/session3-power-user.html)
- [Thermal Data Exercise](https://astrostubbs.github.io/GenAI-for-Scholarship/exercise-thermal.html) - Analyze telescope thermal data
- [Instructor Solutions](https://astrostubbs.github.io/GenAI-for-Scholarship/exercise-thermal-solutions.html)
- [API Setup Guide](https://astrostubbs.github.io/GenAI-for-Scholarship/setting-up-claude-code.html) - For Harvard users

---

## Time and Location

Sessions run 4:00 pm to 5:30 pm in **Northwest Building, Room B103**, followed by a reception and further discussion.

**Address:**
Northwest Building
52 Oxford Street, Cambridge, MA

---

## Repository Structure

```
GenAI-for-Scholarship/
├── index.html                          # Main course website
├── session1-foundation.html            # Session 1: The Basics
├── session2-coder.html                 # Session 2: AI-Empowered Coder
├── session3-power-user.html            # Session 3: Claude Code CLI
├── exercise-session1.html              # Session 1 exercises
├── exercise-thermal.html               # Session 3 thermal data exercise
├── exercise-thermal-solutions.html     # Instructor solutions
├── setting-up-claude-code.html         # Harvard API setup guide
├── notebooks/
│   ├── session1/                       # Session 1 notebooks (if any)
│   ├── session2/
│   │   └── ai_python_demo.ipynb       # AI integration demo notebook
│   └── session3/                       # Session 3 notebooks (if any)
├── data/
│   ├── session1/                       # Session 1 data files
│   ├── session2/                       # Session 2 data files
│   └── session3/
│       ├── rubin_mirror_temps.csv     # Telescope thermal data
│       ├── plot_temperature.py         # Analysis scripts
│       ├── fourier_analysis.py
│       └── ml_sunset_comparison_v2.py  # ML comparison script
├── draft_proposal.pdf                  # Sample NSF proposal (with errors)
├── draft_proposal.tex                  # LaTeX source
├── proposal_errors_reference.txt       # Instructor guide for proposal errors
├── NSF 25-508_ Designing_Materials.pdf # NSF DMREF call
├── nsf23_1.pdf                         # NSF proposal guide
├── style.css                           # Website styling
├── HDSI.png                            # Harvard HDSI logo
├── FAS.png                             # Harvard FAS logo
├── GeminiAccess.png                    # Gemini access screenshot
└── README.md                           # This file
```

---

## Key Features

### Session 1: The Basics
- **Gemini:** General-purpose AI assistant for writing, coding, and analysis
- **NotebookLM:** Document analysis with source-grounded responses
  - Exercise: Compare draft NSF proposal against guidelines
  - Identifies compliance issues with citations
- **Gems:** Custom AI assistants with persistent prompts
- **Ethics Discussion:** Responsible AI use and disclosure practices

### Session 2: The AI-Empowered Coder
Three approaches to AI-integrated Python programming:

1. **Google Colab with Gemini**
   - Built-in AI assistance
   - Debugging, documentation, code generation
   - Demo notebook included

2. **Local Jupyter Notebooks**
   - Harvard HUIT API integration
   - Full control over environment
   - Works with local files

3. **Harvard RC Cluster**
   - High-performance computing
   - GPU access for ML workloads
   - Large-scale data analysis

### Session 3: Claude Code CLI
- **Autonomous AI agent** that plans and executes multi-step tasks
- **Real data analysis:** Telescope thermal data from Vera C. Rubin Observatory
- **Plan Mode:** Review implementation strategy before execution
- **Machine Learning:** Compare 5 ML methods for temperature prediction
- **Harvard API setup:** Secure access through HUIT infrastructure

---

## For Harvard Users

### API Access

To use AI tools beyond the workshop, Harvard affiliates can obtain API keys through HUIT:

📚 **[Complete API Setup Guide](https://astrostubbs.github.io/GenAI-for-Scholarship/setting-up-claude-code.html)**

**Key Steps:**
1. Request HUIT billing number
2. Register your "App" in HUIT API Portal
3. Configure environment variables for Harvard endpoint
4. **Set monthly spending limits** (PI is responsible for costs)

**Important:** API usage is billed to PI accounts. Always coordinate with your advisor and set upper spending limits.

---

## Ethics and Responsible Use

Throughout the course, we emphasize:

- **Disclosure:** Always disclose AI use to collaborators, advisors, journals, and funding agencies
- **Verification:** You remain responsible for all AI-assisted work
- **Research Group Norms:** Establish clear expectations between students, postdocs, and PIs
- **Field-Specific Practices:** Stay current with evolving norms in your discipline
- **Transparency:** When in doubt, err on the side of disclosure

---

## Technical Requirements

### All Sessions
- Laptop (Mac or Windows)
- Harvard-affiliated Google account (e.g., yourname@g.harvard.edu)

### Session 1
- Web browser
- Internet connection

### Session 2
- Python 3.7+
- Google Colab access
- (Optional) Jupyter notebook for local work
- (Optional) Harvard RC account for cluster access

### Session 3
- macOS or Linux (for Claude Code CLI)
- Terminal/command line familiarity
- Python environment

---

## Resources

- **Course Website:** [https://astrostubbs.github.io/GenAI-for-Scholarship/](https://astrostubbs.github.io/GenAI-for-Scholarship/)
- **Harvard RC Documentation:** [https://docs.rc.fas.harvard.edu](https://docs.rc.fas.harvard.edu)
- **Claude Code Documentation:** [https://code.claude.com/docs](https://code.claude.com/docs)
- **Anthropic API Documentation:** [https://docs.anthropic.com](https://docs.anthropic.com)

---

## License

© 2026 President and Fellows of Harvard College.

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

**Attribution:** This material is based on work by Christopher Stubbs and the Harvard Data Science Initiative, with contributions from the Faculty of Arts and Sciences.

---

## Contact

For questions about the workshop:
- Harvard Data Science Initiative: [https://datascience.harvard.edu](https://datascience.harvard.edu)
- Course materials issues: [GitHub Issues](https://github.com/astrostubbs/GenAI-for-Scholarship/issues)

---

## Acknowledgments

Materials developed with assistance from Claude (Anthropic) for code generation, documentation, and curriculum design.

Workshop support provided by:
- Harvard Data Science Initiative (HDSI)
- Faculty of Arts and Sciences (FAS)
- Harvard Research Computing (RC)
- Harvard University Information Technology (HUIT)

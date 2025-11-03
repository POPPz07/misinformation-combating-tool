## Overview

*TruthLens - Misinformation Combating Tool* is an open-source platform designed to help users identify, analyze, and combat misinformation online. Leveraging both automated and manual review processes, it provides features for fact-checking, source validation, and reporting suspicious content.

## Features

- *Automated Misinformation Detection:* Uses natural language processing and rule-based algorithms to flag potential misinformation.
- *Fact-Checking Dashboard:* Allows users to review flagged content, check sources, and see fact-check summaries.
- *Source Validation:* Evaluates the credibility of linked sources and websites.
- *User Reporting:* Users can report misinformation for further analysis and review.
- *Visualization:* Interactive charts and visualizations to track misinformation trends.
- *Multi-Language Support:* Built primarily in JavaScript, with Python components for backend analysis.

## Tech Stack

- *Frontend:* JavaScript, HTML, CSS
- *Backend:* Python (for NLP and analysis)

## Getting Started

### Prerequisites

- Node.js & npm (for frontend)
- Python 3.x (for backend)

### Installation

1. *Clone the repository*
   bash
   git clone https://github.com/POPPz07/misinformation-combating-tool.git
   cd misinformation-combating-tool
   

2. *Install frontend dependencies*
   bash
   npm install
   

3. *Install backend dependencies*
   bash
   pip install -r requirements.txt
   

4. *Run the application*
   - Start frontend:
     bash
     npm start
     
   - Start backend:
     bash
     python backend/app.py
     

5. *Access the dashboard*
   Open your browser and navigate to http://localhost:3000 (or the configured port).

## Usage

- Upload or paste content to analyze for misinformation.
- Review flagged items and fact-check summaries.
- Report suspicious content or sources.
- View visual trends and summaries.

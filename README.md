# After-Call Automation Platform

## Overview

Kalam After-Call Automation Platform is an AI-powered pipeline that automates post-call documentation for customer support operations.

The platform processes call transcripts, enriches them with customer information and handbook knowledge, generates structured documentation using a Large Language Model, applies business rules, determines review decisions, stores outputs, evaluates prediction quality against a ground truth dataset, and generates analytical dashboards.

---

## Features

- Transcript ingestion
- Customer lookup
- Handbook retrieval
- AI documentation generation
- Business rule engine
- Human review routing
- JSON output storage
- Automatic evaluation
- Performance dashboard
- Charts and analytics

---

## Project Structure

app/

data/

outputs/

tests/

README.md

requirements.txt

---

## Pipeline

Transcript

↓

Customer Lookup

↓

Handbook Search

↓

Documentation Engine

↓

Business Rules

↓

Review Decision

↓

Storage

↓

Evaluation

↓

Dashboard

---

## Technologies

Python

Google Gemini

LangChain

ChromaDB

RapidFuzz

Matplotlib

JSON

---

## Installation

```bash
pip install -r requirements.txt
